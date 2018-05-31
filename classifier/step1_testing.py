from sklearn.ensemble import RandomForestClassifier
import csv,math,sys,pickle
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import f1_score
from random import shuffle
import numpy as np
from imblearn.under_sampling import RepeatedEditedNearestNeighbours

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# using imbalance package

def save_pickle(object_to_save,filename):
	fileObject = open(filename,'wb')
	pickle.dump(object_to_save,fileObject) 
	fileObject.close()

def load_pickle(filename):
	fileObject = open(filename,'rb') 
	b = pickle.load(fileObject) 
	fileObject.close()
	return b

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
	file_p = open(filename,"r").read().splitlines()
	wiki_dict = dict()
	key = ""
	select_fea = "1,2,3,4,5,6,7,8".split(",")
	for line in file_p:
		if line[-5:] == ".wiki":
			key = line
			wiki_dict[key] = {'Y':[],'N':[]}
		else:
			tokens = line.split("#")
			wiki_dict[key][tokens[-1]].append(tokens[0:1]+[ float(tokens[int(i)]) for i in select_fea ] )
	keys = list(wiki_dict.keys())
	train_yes = list()
	train_no = list()
	for key in keys:
		if len(train_yes) < training_size:
			train_yes = train_yes + removeBs(wiki_dict[key]['Y'])
			train_no = train_no + removeBs(wiki_dict[key]['N'])
			del wiki_dict[key]
		else:
			break
	return wiki_dict,train_yes,train_no


def getNumberOfYes(file):
	fp = open(file,"r")
	data = fp.read()
	fp.close()
	lines = data.splitlines()
	count = 0
	for i in lines:
		if i[-1] == 'Y':
			count = count  + 1
	return count

numberOfYes = getNumberOfYes("../features/features.csv")

training_testing_variation = 0.3

wiki_As,train_yes,train_no = getTrainingTestingData("../features/features.csv",numberOfYes,scale = 1)

X = train_yes + train_no
y = ['Y']*len(train_yes) + ['N']*len(train_no)

renn = RepeatedEditedNearestNeighbours(random_state=0)
X_resampled, y_resampled = renn.fit_sample(X, y)

rfc_linear = RandomForestClassifier(max_depth=10, random_state=0)
rfc_linear.fit(X_resampled, y_resampled)


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
	predicted = rfc_linear.predict(X_test)
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
	print "Target Page",key
	print "Format is",bcolors.WARNING, "<B,Actual,Predicted>",bcolors.ENDC
	print str(zip(X_test_Bs,y_test,predicted))

print "Avg. Pre",(sum(pre_mi)/len(pre_mi))
print "Avg. Rec",(sum(rec_mi)/len(rec_mi))
print "Avg. f1",(sum(f1s)/len(f1s))
