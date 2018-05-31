from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from imblearn.under_sampling import NearMiss
from sklearn import svm
from sklearn.svm import LinearSVC
import csv,math,sys,pickle,os
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import f1_score
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE, ADASYN
from random import shuffle
import numpy as np
# import matplotlib.pyplot as plt
from imblearn.under_sampling import ClusterCentroids
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import RepeatedEditedNearestNeighbours
from imblearn.under_sampling import AllKNN
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.under_sampling import CondensedNearestNeighbour
from imblearn.under_sampling import OneSidedSelection
from imblearn.under_sampling import NeighbourhoodCleaningRule
from imblearn.under_sampling import InstanceHardnessThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
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
	select_fea = sys.argv[1].split(",")
	for line in file_p:
		if line[-5:] == ".wiki":
			key = line
			wiki_dict[key] = {'Y':[],'N':[]}
		else:
			tokens = line.split("#")
			wiki_dict[key][tokens[-1]].append(tokens[0:1]+[ float(tokens[int(i)]) for i in select_fea ] )
	keys = list(wiki_dict.keys())
	# Random-ness
	# shuffle(keys)
	# in_degree#out_degree#tfidf#in_sent#out_sent#class
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

number_of_yes = 11537
number_of_no = 100887

training_testing_variation = 0.3

wiki_As,train_yes,train_no = getTrainingTestingData("features_phy1.csv",number_of_yes,scale = 1)

print "No. yes",len(train_yes)
print "No. no",len(train_no)
print "Test set of A's",len(wiki_As)
X = train_yes + train_no
y = ['Y']*len(train_yes) + ['N']*len(train_no)

save_pickle(wiki_As,"wiki_As")
save_pickle(X,"X")
save_pickle(y,"y")


if os.path.isfile("X_phy_features_renn_"+"".join(sys.argv[1].split(','))):
	X_resampled = load_pickle("X_phy_features_renn_"+"".join(sys.argv[1].split(',')))
	y_resampled = load_pickle("y_phy_features_renn_"+"".join(sys.argv[1].split(',')))
else:
	renn = RepeatedEditedNearestNeighbours(random_state=0)
	X_resampled, y_resampled = renn.fit_sample(X, y)
	save_pickle(X_resampled,"X_phy_features_renn_"+"".join(sys.argv[1].split(',')))
	save_pickle(y_resampled,"y_phy_features_renn_"+"".join(sys.argv[1].split(',')))



print "Done Sampling"
print "After sampling total",len(X_resampled)

rfc = RandomForestClassifier(max_depth=10, random_state=0)
rfc.fit(X_resampled, y_resampled)

print "Number of training ",len(X_resampled)



pre_mi = []
rec_mi = []
f1s = []
roc_auc = []
keys = list(wiki_As.keys())
pre_reca_dict = dict()
# no_wikilinks = getWikilinksNo()
mymax = 0
loop_count = 0
for key in keys:
	print "Key :",key,len(keys)-loop_count
	loop_count = loop_count+1
	if len(wiki_As[key]['Y']) == 0 or len(wiki_As[key]['N']) == 0:
		continue
	number_yes = len(wiki_As[key]['Y'])
	number_no = len(wiki_As[key]['N'])
	X_test = removeBs(wiki_As[key]['Y']) + removeBs(wiki_As[key]['N'])
	X_test_Bs = removeFeatures(wiki_As[key]['Y']) + removeFeatures(wiki_As[key]['N'])
	y_test = ['Y']*number_yes + ['N']*number_no
	predicted = rfc.predict(X_test)
	cnf_matrix = confusion_matrix(y_test, predicted)
	report = classification_report(y_test, predicted)
	y_pred = predicted
	
	a = precision_score(y_test, list(predicted),pos_label='Y', average='binary')
	b = recall_score(y_test, list(predicted),pos_label='Y', average='binary')
	c = f1_score(y_test, list(predicted),pos_label='Y', average='binary')
	pre_mi.append(a)
	rec_mi.append(b)
	f1s.append(c+0.01)
	# pre_reca_dict[key] = [a,b,no_wikilinks[key]]
	# mymax = max(mymax,no_wikilinks[key])

print "Avg. Pre",(sum(pre_mi)/len(pre_mi))
print "Avg. Rec",(sum(rec_mi)/len(rec_mi))
print "Avg. f1",(sum(f1s)/len(f1s))



# cc = ClusterCentroids(random_state=0)
# X_resampled, y_resampled = cc.fit_sample(X, y)
# ros = RandomOverSampler()
# X_resampled, y_resampled = ros.fit_sample(X_train,y_train)
# rus = RandomUnderSampler(random_state=0)
# X_resampled, y_resampled = rus.fit_sample(X_train, y_train)
# nm1 = NearMiss(random_state=0, version=2)
# X_resampled, y_resampled = nm1.fit_sample(X, y)
# renn = RepeatedEditedNearestNeighbours(random_state=0)
# X_resampled, y_resampled = renn.fit_sample(X, y)
# save_pickle(X_resampled,"X_resampled_renn_9_1234567")
# save_pickle(y_resampled,"y_resampled_renn_9_1234567")
# X_resampled = load_pickle("X_resampled_renn_9_1234567")
# y_resampled = load_pickle("y_resampled_renn_9_1234567")
# allknn = AllKNN(random_state=0)
# X_resampled, y_resampled = allknn.fit_sample(X, y)
# enn = EditedNearestNeighbours(random_state=0)
# X_resampled, y_resampled = enn.fit_sample(X, y)
# cnn = CondensedNearestNeighbour(random_state=0)
# X_resampled, y_resampled = cnn.fit_sample(X, y)
# ncr = NeighbourhoodCleaningRule(random_state=0)
# X_resampled, y_resampled = ncr.fit_sample(X, y)
# oss = OneSidedSelection(random_state=0)
# X_resampled, y_resampled = oss.fit_sample(X, y)
# iht = InstanceHardnessThreshold(random_state=0,estimator=LogisticRegression())
# X_resampled, y_resampled = iht.fit_sample(X, y)
# smote_enn = SMOTEENN(random_state=0)
# X_resampled, y_resampled = smote_enn.fit_sample(X, y)
# smote_tomek = SMOTETomek(random_state=0)
# X_resampled, y_resampled = smote_tomek.fit_sample(X, y)
# X_resampled, y_resampled = SMOTE().fit_sample(X, y)
# X_resampled, y_resampled = ADASYN().fit_sample(X, y)
# X_resampled, y_resampled = SMOTE(kind='borderline1').fit_sample(X, y)
# smote_enn = SMOTEENN(random_state=0)
# X_resampled, y_resampled = smote_enn.fit_sample(X, y)