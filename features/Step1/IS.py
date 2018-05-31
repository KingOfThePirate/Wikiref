import os,sys,pickle
from sklearn.metrics import jaccard_similarity_score

def save_pickle(object_to_save,filename):
	fileObject = open(filename,'wb')
	pickle.dump(object_to_save,fileObject) 
	fileObject.close()

def load_pickle(filename):
	fileObject = open(filename,'rb') 
	b = pickle.load(fileObject) 
	fileObject.close()
	return b

def parseAB(text):
	if len(text)==0 or text is None:
		return None.None,None
	lines = text.splitlines()
	a_name = lines[0][2:].split("##__##")[0]
	bs = list()
	y_n = list()
	for line in lines:
		b = line.split("##__##")[1].rsplit(":",1)[0]
		y_n.append(line.split("##__##")[1].split(":")[-1][0])
		bs.append(b)
	return [a_name,bs,y_n]

def getJaccard(x,y):
	if x is None or y is None or (len(x) == 0 and len(y) == 0):
		return 0
	intersection_cardinality = len(set.intersection(*[set(x), set(y)]))
	union_cardinality = len(set.union(*[set(x), set(y)]))
	try:
		return round(intersection_cardinality/float(union_cardinality),3)
	except ZeroDivisionError:
		return 0


AB_overlap_relation = open("../../prepocessing/AB_overlap_relation","r").read().split("@@New@@\n")[1:]
AB_inlink_dict = load_pickle("../../pickle_files/AB_inlink_dict")
AB_relation_dict = load_pickle("../../pickle_files/AB_relation_dict")
in_degree_feature = open("in_degree_feature","w")

for AB in AB_overlap_relation:
	A,Bs,YN = parseAB(AB)
	yes = list()
	no = list()
	in_degree_feature.write("@@"+A+"@@\n")

	for x in range(len(Bs)):
		B = Bs[x]
		try:
			indegree_simi = getJaccard(AB_inlink_dict[A],AB_inlink_dict[B])
		except KeyError:
			print "KeyError",A,B
			continue

		if YN[x] == "Y":
			in_degree_feature.write(B+"##Y##"+str(indegree_simi)+"\n")
			yes.append(indegree_simi)
		else:
			in_degree_feature.write(B+"##N##"+str(indegree_simi)+"\n")
			no.append(indegree_simi)
	yes.sort()
	no.sort()

	if len(yes) != 0:
		in_degree_feature.write("$$$$Result_Y$$$$\n")
		in_degree_feature.write("Min:"+str(yes[0])+"\n")
		in_degree_feature.write("Max:"+str(yes[-1])+"\n")
		in_degree_feature.write("Median:"+str(yes[len(yes)/2])+"\n")
		in_degree_feature.write("Mean:"+str(round(sum(yes)/float(len(yes)),3))+"\n")
	if len(no) != 0:
		in_degree_feature.write("$$$$Result_N$$$$\n")
		in_degree_feature.write("Min:"+str(no[0])+"\n")
		in_degree_feature.write("Max:"+str(no[-1])+"\n")
		in_degree_feature.write("Median:"+str(no[len(no)/2])+"\n")
		in_degree_feature.write("Mean:"+str(round(sum(no)/float(len(no)),3))+"\n")
	print "Done",A
in_degree_feature.close()