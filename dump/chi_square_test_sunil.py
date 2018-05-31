# import pandas
import numpy,math
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.model_selection import train_test_split

def getXY(file_name):
	X = list()
	Y = list()
	csvfile = open(file_name, 'r').read().splitlines()
	feature_name = csvfile[0].split(',')

	for row in csvfile[1:]:
		if row=='':
			continue
		row = row.split(",")
		X.append([abs(float(i)) for i in row[:34]]+[abs(float(i)) for i in  row[35:-1]])
		Y.append(row[-1])
		# if row[-1] == '':
			# print row[-1]
	return X,Y,feature_name
def my_train_test_split(X_no,y_no,train_size,scale = 1):
	training_size = int(math.ceil(train_size*(1.0-training_testing_variation)))
	testing_size = train_size - training_size
	testing_size = testing_size*scale
	return [ X_no[0:training_size] , X_no[training_size:training_size+testing_size] , y_no[0:training_size] , y_no[training_size:training_size+testing_size] ]

def takeSecond(elem):
    return elem[1]

X,y,feature_name = getXY("/home/pranjal/Downloads/out.csv")
feature_name = feature_name[:34] + feature_name[35:-1]
labels = ["running", "walking", "cycling", "sitting", "sleeping", "staircase", "eating"]
y = [ labels.index(i)+1 for i in y ]

X = numpy.asarray(X,dtype = 'f')

test = SelectKBest(score_func=chi2, k=10)
fit = test.fit(X,y)

# summarize scores
numpy.set_printoptions(precision=2)
score_featureName = [(i,fit.scores_[i]) for i in range(len(fit.scores_))]

score_featureName = sorted(score_featureName, key=takeSecond, reverse=True)

for i,j in score_featureName:
	print feature_name[i],j