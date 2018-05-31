import os
from random import shuffle

def getTrainingTestingData(filename,train_size,scale = 1):
	training_size = int(math.ceil(train_size*(1.0-training_testing_variation)))
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
	shuffle(keys)
	train_yes = list()
	train_no = list()
	for key in keys:
		if len(train_yes) == training_size:
			break
		train_yes.append(wiki_dict[key]['Y'])
		train_no.append(wiki_dict[key]['N'])
		del wiki_dict[key]
	#Random-ness
	shuffle(train_no)
	train_no = train_no[0:training_size]
	return wiki_dict,train_yes,train_no

training_testing_variation = 0.3