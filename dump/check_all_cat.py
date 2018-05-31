from collections import defaultdict
import pickle,sys

def save_pickle(object_to_save,filename):
	fileObject = open(filename,'wb')
	pickle.dump(a,fileObject) 
	fileObject.close()

def load_pickle(filename):
	fileObject = open(filename,'rb') 
	b = pickle.load(fileObject) 
	fileObject.close()
	return b

cat_to_id = load_pickle("cat_to_id")

cat_file = open("all_category_uniq_1","r").read().splitlines()
# cat_file_r = open("all_category_uniq_1","w")

for cat in cat_file:
	try:
		a = cat_to_id[ cat ]
		# cat_file_r.write(cat+"\n")
	except KeyError:
		print cat