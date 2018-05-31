from sklearn import svm
import csv,math
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_fscore_support
from random import shuffle
import numpy as np
import matplotlib.pyplot as plt
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
			wiki_dict[key][tokens[-1]].append(tokens[0:-1])
	keys = list(wiki_dict.keys())
	#Random-ness
	# shuffle(keys)
	# in_degree#out_degree#tfidf#in_sent#out_sent#class
	train_yes = list()
	train_no = list()
	for key in keys:
		if len(train_yes) >= training_size:
			break
		train_yes = train_yes + wiki_dict[key]['Y']
		train_no = train_no + wiki_dict[key]['N']
		del wiki_dict[key]
	#Random-ness
	shuffle(train_no)
	train_no = train_no[0:training_size]
	return wiki_dict,train_yes,train_no
def getXY(file_name):
	X = list()
	Y = list()
	csvfile = open(file_name, 'r').read().splitlines()
	# Randomness
	shuffle(csvfile)
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

wiki_As,train_yes,train_no = getTrainingTestingData("features2.csv",len(X_yes),scale = 1)
X_train = train_yes + train_no
y_train = ['Y']*len(train_yes) + ['N']*len(train_no)
# Linear Kernel
svc_linear = svm.SVC(kernel='linear', C = 1.0)
svc_linear.fit(X_train, y_train)
print "Number of training ",len(X_train)
# file_w = open("for_each_A","w")


pre_ma = []
rec_ma = []
pre_bi = []
rec_bi = []
pre_mi = []
rec_mi = []
roc_auc = []
keys = list(wiki_As.keys())
pre_reca_dict = dict()
no_wikilinks = getWikilinksNo()
mymax = 0
for key in keys:
	# print key
	if len(wiki_As[key]['Y']) == 0 or len(wiki_As[key]['N']) == 0:
		continue
	number_yes = len(wiki_As[key]['Y'])
	number_no = len(wiki_As[key]['N'])
	X_test = wiki_As[key]['Y']+wiki_As[key]['N']
	y_test = ['Y']*number_yes + ['N']*number_no
	predicted= svc_linear.predict(X_test)
	cnf_matrix = confusion_matrix(y_test, predicted)
	report = classification_report(y_test, predicted)
	y_pred = predicted
	
	a = precision_score(y_test[0:number_yes], y_pred[0:number_yes])
	b = recall_score(y_test[0:number_yes], y_pred[0:number_yes])
	pre_mi.append(a)
	rec_mi.append(b)

	pre_reca_dict[key] = [a,b,no_wikilinks[key]]
	mymax = max(mymax,no_wikilinks[key])
	# roc_auc.append(roc_auc_score(y_test, y_scores))

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
print bucket_graph

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

plt.show()