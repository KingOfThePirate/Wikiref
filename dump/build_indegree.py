import os,sys,pickle

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
		return None,[]
	lines = text.splitlines()
	A = lines[0][2:].split("##__##")[0]
	Bs = list()
	YN = list()
	for line in lines:
		Bs.append(line.split("##__##")[1].rsplit(":",1)[0])
		YN.append(line.split("##__##")[1].rsplit(":",1)[-1])
	return A,Bs,YN

As_fp = open("../../wiki_pages_in_phy_5_As_geq05","r")
As = As_fp.read().splitlines()
As_fp.close()

AB_relation_fp = open("../../AB_overlap_relation_geq05","r")
AB_relation = AB_relation_fp.read().split("@@New@@\n")[1:]
AB_relation_fp.close()

#Key is an A and value is 2 element tuple where first element is Bs and 2nd is their corresponding relationship
AB_relation_dict = dict()

ALL_As_Bs = list()

for each in AB_relation:
	A,Bs,YN = parseAB(each)
	ALL_As_Bs.append(A)
	ALL_As_Bs.extend(Bs)
ALL_As_Bs = list(set(ALL_As_Bs))
save_pickle(ALL_As_Bs,"../../ALL_As_Bs")