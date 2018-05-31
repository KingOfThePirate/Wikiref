import pandas
import numpy,math
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.model_selection import train_test_split

def getXY(file_name):
	X = list()
	Y = list()
	csvfile = open(file_name, 'r').read().splitlines()
	# Randomness
	# in_degree#out_degree#tfidf#in_sent#out_sent#class
	for row in csvfile:
		if row[-5:] == ".wiki":
			continue
		row = row.split("#")
		one_x = list()
		for x1 in row[1:-1]:
			one_x.append(abs(float(x1)))
		X.append(one_x)
		Y.append(row[-1])
	return [X,Y]
def my_train_test_split(X_no,y_no,train_size,scale = 1):
	training_size = int(math.ceil(train_size*(1.0-training_testing_variation)))
	testing_size = train_size - training_size
	testing_size = testing_size*scale
	return [ X_no[0:training_size] , X_no[training_size:training_size+testing_size] , y_no[0:training_size] , y_no[training_size:training_size+testing_size] ]

"tfidf_analysis","out_sent_analysis","in_sent_analysis","out_degree_analysis","in_degree_analysis","summ_VecEmb_analysis","out_link_sent2vec_analysis"
X,y = getXY("features9.csv")
y = [1.0 if x=='Y' else 0.0 for x in y ]
# print y
# feature extraction
test = SelectKBest(score_func=chi2, k=7)
fit = test.fit(X,y)

# summarize scores
numpy.set_printoptions(precision=2)
print([round(i,3) for i in fit.scores_])
