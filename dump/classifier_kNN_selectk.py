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
from numpy import array
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

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
			wiki_dict[key][tokens[-1]].append([float(i) for i in tokens[0:-1]])
	keys = list(wiki_dict.keys())
	#Random-ness
	# shuffle(keys)
	# in_degree#out_degree#tfidf#in_sent#out_sent#class
	train_yes = list()
	train_no = list()
	for key in keys:
		if len(train_yes) < training_size and len(train_no) < training_size:
			train_yes = train_yes + wiki_dict[key]['Y']
			train_no = train_no + wiki_dict[key]['N']
			del wiki_dict[key]
		elif len(train_yes) < training_size :
			train_yes = train_yes + wiki_dict[key]['Y']
			del wiki_dict[key]
		else:
			break
	#Random-ness
	shuffle(train_no)
	# train_no = train_no[0:training_size]
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


X_no,y_no = getXY("features_no.csv")
X_yes,y_yes = getXY("features_yes.csv")

training_testing_variation = 0.3

wiki_As,train_yes,train_no = getTrainingTestingData("features2.csv",len(X_yes),scale = 1)

print "No. yes",len(train_yes)
print "No. no",len(train_no)
print "Test set of A's",len(wiki_As)
X_train = train_yes + train_no
y_train = ['Y']*len(train_yes) + ['N']*len(train_no)

for k in xrange(1,100,2):
	kNN_clf = KNeighborsClassifier(n_neighbors=k)
	kNN_clf.fit(X_train, y_train)

	pre_mi = []
	rec_mi = []

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
		predicted = kNN_clf.predict(X_test)
		cnf_matrix = confusion_matrix(y_test, predicted)
		report = classification_report(y_test, predicted)
		y_pred = predicted
		
		a = precision_score(y_test, list(y_pred),pos_label='Y', average='binary')
		b = recall_score(y_test, list(y_pred),pos_label='Y', average='binary')
		pre_mi.append(a)
		rec_mi.append(b)

		pre_reca_dict[key] = [a,b,no_wikilinks[key]]
		mymax = max(mymax,no_wikilinks[key])

	print "Avg. Pre",(sum(pre_mi)/len(pre_mi)),k
	print "Avg. Rec",(sum(rec_mi)/len(rec_mi)),k