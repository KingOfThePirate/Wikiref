from sklearn import svm
import csv,math,sys,pickle,urllib
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
from sklearn.ensemble import RandomForestClassifier


# PCA, Adding more features

def save_pickle(object_to_save,filename):
	fileObject = open(filename,'wb')
	pickle.dump(object_to_save,fileObject) 
	fileObject.close()

def load_pickle(filename):
	fileObject = open(filename,'rb') 
	b = pickle.load(fileObject) 
	fileObject.close()
	return b

def listDiff(a,b):
	return [ abs(a[i]-b[i]) for i in range(len(a)) ]

def listProd(a,b):
	return [ a[i]*b[i] for i in range(len(a)) ]

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
	# ["tfidf_analysis","out_sent_analysis","in_sent_analysis","out_degree_analysis","in_degree_analysis","CN_simi_analysis_meanofmean","CN_simi_PL_analysis_meanofmean","CN_simi_PL_analysis_meanofmin"]
	select_fea = sys.argv[1].split(",")
	loop_check = False
	for line in file_p:
		if line[-5:] == ".wiki":
			key = line
			try:
				As_vec = wiki_vector_ConVec_Heuristic[wiki_name_to_wiki_id[urllib.unquote(key).decode('utf8')]]
				As_vec = list(As_vec)
				loop_check = True
				wiki_dict[key] = {'Y':[],'N':[]}
			except KeyError:
				print "A ",key
				loop_check = False
		elif loop_check:
			tokens = line.split("#")
			try:
				Bs_vec = wiki_vector_ConVec_Heuristic[wiki_name_to_wiki_id[urllib.unquote(tokens[0]).decode('utf8')]]
				Bs_vec = list(Bs_vec)
			except KeyError:
				print "B ",key
				continue
			wiki_dict[key][tokens[-1]].append(tokens[0:1]+[ float(tokens[int(i)]) for i in select_fea ] + As_vec + Bs_vec + listDiff(As_vec,Bs_vec) + listProd(As_vec,Bs_vec))
	keys = list(wiki_dict.keys())
	print "Total A's",len(keys)
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
wiki_id_to_wiki_name = load_pickle("wiki_id_to_wiki_name")
wiki_name_to_wiki_id = load_pickle("wiki_name_to_wiki_id")
wiki_vector_ConVec_Heuristic = load_pickle(sys.argv[2])

training_testing_variation = 0.3

wiki_As,train_yes,train_no = getTrainingTestingData("features7.csv",10000,scale = 1)

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
