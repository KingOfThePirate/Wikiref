import pickle,os,sys


def save_pickle(object_to_save,filename):
	fileObject = open(filename,'wb')
	pickle.dump(object_to_save,fileObject) 
	fileObject.close()

def load_pickle(filename):
	fileObject = open(filename,'rb') 
	b = pickle.load(fileObject) 
	fileObject.close()
	return b

def get_depth_lca(a,b):
	count = 0
	for i in range(min(len(a),len(b))):
		if a[i] == b[i]:
			count += 1
		else:
			break
	return count

paths_from_root_to_all_nodes = load_pickle("paths_from_root_to_all_nodes")

paths_A = paths_from_root_to_all_nodes["Natural_language_processing"]
paths_B = paths_from_root_to_all_nodes["Information_retrieval"]

print paths_A
print "****************"
print paths_B
depth_lca_max = 0.0
a_max = 0.0
b_max = 0.0
for a in paths_A:
	for b in paths_B:
		depth_lca = get_depth_lca(a,b)
		if depth_lca > depth_lca_max:
			depth_lca_max = depth_lca
			a_max = a
			b_max = b

print "***********"
print a_max
print b_max
print depth_lca_max
print (2.0*depth_lca_max)/(len(a_max)+len(b_max))