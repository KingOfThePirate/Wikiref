from sklearn import svm
import csv,math,sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import f1_score
from random import shuffle
import numpy as np
import matplotlib.pyplot as plt

def removeBs(a):
	ret = list()
	bs = list()
	for i in a:
		ret.append(i[1:])
		bs.append(i[0])
	return ret

def removeFeatures(a):
	ret = list()
	bs = list()
	for i in a:
		ret.append(i[1:])
		bs.append(i[0])
	return bs


def getWikilinksNo():
	filename = "count_3_3.csv"
	with open(filename, 'r') as csvfile:
		# creating a csv reader object
		csvreader = csv.reader(csvfile, delimiter = ';')
		# extracting field names through first row
		fields = csvreader.next()
		pre_reca_dict = dict()
		# extracting each data row one by one
		for row in csvreader:
			if "TLDR.wiki" in row:
				pre_reca_dict['TL;DR.wiki'] = 71
			else:
				pre_reca_dict[row[0]] = int(row[5])
	return pre_reca_dict


def getTrainingTestingData(filename,train_size,scale = 1):
	training_size = int(math.ceil(train_size*(1.0-training_testing_variation)))
	# print training_size
	file_p = open(filename,"r").read().splitlines()
	wiki_dict = dict()
	key = ""
	for line in file_p:
		if line[-5:] == ".wiki":
			key = line
			wiki_dict[key] = {'Y':[],'N':[]}
		else:
			tokens = line.split("#")
			wiki_dict[key][tokens[-1]].append(tokens[0:1]+[float(i) for i in tokens[1:-1]])
	keys = list(wiki_dict.keys())
	# Random-ness
	# shuffle(keys)
	# in_degree#out_degree#tfidf#in_sent#out_sent#class
	train_yes = list()
	train_no = list()
	for key in keys:
		if len(train_yes) < training_size and len(train_no) < training_size:
			train_yes = train_yes + removeBs(wiki_dict[key]['Y'])
			train_no = train_no + removeBs(wiki_dict[key]['N'][0:len(wiki_dict[key]['Y'])])
			del wiki_dict[key]
		elif len(train_yes) < training_size :
			train_yes = train_yes + removeBs(wiki_dict[key]['Y'])
			del wiki_dict[key]
		else:
			break
	# Random-ness
	# shuffle(train_no)
	# train_no = train_no[0:training_size]
	return wiki_dict,train_yes,train_no

def getXY(file_name):
	X = list()
	Y = list()
	csvfile = open(file_name, 'r').read().splitlines()
	# Randomness
	# shuffle(csvfile)
	for row in csvfile:
		row = row.split("#")
		one_x = list()
		for x1 in row[0:-1]:
			one_x.append(float(x1))
		X.append(one_x)
		Y.append(row[-1])
	return [X,Y]
def my_train_test_split(X_no,y_no,train_size,scale = 1):
	training_size = int(math.ceil(train_size*(1.0-training_testing_variation)))
	testing_size = train_size - training_size
	testing_size = testing_size*scale
	return [ X_no[0:training_size] , X_no[training_size:training_size+testing_size] , y_no[0:training_size] , y_no[training_size:training_size+testing_size] ]


# 14608(0.7,yes) 14895(0.047,no)
X_no,y_no = getXY("features_no.csv")
X_yes,y_yes = getXY("features_yes.csv")

training_testing_variation = 0.3

wiki_As,train_yes,train_no = getTrainingTestingData("features6.csv",len(X_yes),scale = 1)

print "No. yes",len(train_yes)
print "No. no",len(train_no)
print "Test set of A's",len(wiki_As)
X_train = train_yes + train_no
y_train = ['Y']*len(train_yes) + ['N']*len(train_no)
# Linear Kernel
svc_linear = svm.SVC(kernel='linear', C = 1.0)
svc_linear.fit(X_train, y_train)
print "Number of training ",len(X_train)
# file_w = open("for_each_A","w")


pre_mi = []
rec_mi = []
f1s = []
roc_auc = []
keys = list(wiki_As.keys())
pre_reca_dict = dict()
no_wikilinks = getWikilinksNo()
mymax = 0
for key in keys:
	if len(wiki_As[key]['Y']) == 0 or len(wiki_As[key]['N']) == 0:
		continue
	number_yes = len(wiki_As[key]['Y'])
	number_no = len(wiki_As[key]['N'])
	X_test = removeBs(wiki_As[key]['Y']) + removeBs(wiki_As[key]['N'])
	X_test_Bs = removeFeatures(wiki_As[key]['Y']) + removeFeatures(wiki_As[key]['N'])
	y_test = ['Y']*number_yes + ['N']*number_no
	predicted = svc_linear.predict(X_test)
	# print len(predicted),len(y_test)
	cnf_matrix = confusion_matrix(y_test, predicted)
	report = classification_report(y_test, predicted)
	y_pred = predicted
	
	a = precision_score(y_test, list(predicted),pos_label='Y', average='binary')
	b = recall_score(y_test, list(predicted),pos_label='Y', average='binary')
	c = f1_score(y_test, list(predicted),pos_label='Y', average='binary')
	pre_mi.append(a)
	rec_mi.append(b)
	f1s.append(c+0.01)
	pre_reca_dict[key] = [a,b,no_wikilinks[key]]
	mymax = max(mymax,no_wikilinks[key])

print "Avg. Pre",(sum(pre_mi)/len(pre_mi))
print "Avg. Rec",(sum(rec_mi)/len(rec_mi))
print "Avg. f1",(sum(f1s)/len(f1s))


"""
buckets = 10

bucket_size = 50

buckets = [bucket_size]*10
for i in range(1,10):
	buckets[i] = buckets[i-1] + buckets[i]

bucket_graph = [[] for _ in range(0,10)]

for key,value in pre_reca_dict.iteritems():
	bucket = 9
	for i in range(0,9):
		if value[2] < buckets[i]:
			bucket = i
			break
	bucket_graph[bucket].append([round(value[0],3),round(value[1],3)])

pre_bar = list()
recall_bar = list()

for i in range(0,10):
	pre_bar.append(sum([ j[0] for j in bucket_graph[i] ])/(len(bucket_graph[i])*1.0))
	recall_bar.append(sum([ j[1] for j in bucket_graph[i] ])/(len(bucket_graph[i])*1.0))


plt.rcParams.update({'font.size': 22})
fig = plt.figure()
ax = fig.add_subplot(111)

N = 10

ind = np.arange(N)                # the x locations for the groups
width = 0.35                      # the width of the bars

## the bars
rects1 = ax.bar(ind, pre_bar, width,
                color='green',
                error_kw=dict(elinewidth=2,ecolor='red'))

rects2 = ax.bar(ind+width, recall_bar, width,
                    color='red',
                    error_kw=dict(elinewidth=2,ecolor='green'))

# axes and labels
ax.set_xlim(-width,len(ind)+width)
#ax.set_ylim(0,45)
ax.set_ylabel('Average Presicion and Recall')
ax.set_xlabel('Buckets for number of wikilinks in an A ')
ax.set_title('Two bar Graph for average Presicion and Recall')
xTickMarks = ['Bin '+str(i) for i in buckets]
ax.set_xticks(ind+width)
xtickNames = ax.set_xticklabels(xTickMarks)
plt.setp(xtickNames, rotation=45, fontsize=10)

## add a legend
ax.legend( (rects1[0], rects2[0]), ('Presicion', 'Recall') )

plt.show()"""