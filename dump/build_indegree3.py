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

def getWikiFileFormat(name):
	name = name.replace(" ","_")
	name = name.split("#",1)[0]
	name = name.replace("/","$$$")
	if len(name)<5 or name[-5:] != ".wiki":
		name = name + ".wiki"
	return name
def getIntraLinks(file_name,path):
	file_lines_fp = open(path+file_name,"r")
	file_lines = file_lines_fp.read().splitlines()
	file_lines_fp.close()
	try:
		intralinks_index = file_lines.index("##IntraLinks:")
		external_links_index = file_lines.index("##External References Links:")
	except ValueError:
		return []
	intra = file_lines[intralinks_index+1:external_links_index]
	for i in range(len(intra)):
		intra[i] = getWikiFileFormat(intra[i])
	return intra



AB_intra_dict = load_pickle("../../AB_intra_dict")

AB_inlink_dict = dict()

all_keys_inlink_dict = list()

for key in AB_intra_dict:
	all_keys_inlink_dict.extend(AB_intra_dict[key])
all_keys_inlink_dict = list(set(all_keys_inlink_dict))

print "Done"

for key in all_keys_inlink_dict:
	AB_inlink_dict[key] = list()

print "Done"

print len(all_keys_inlink_dict)
temp = len(AB_intra_dict)

i = 0
for key in AB_intra_dict:
	for out_link in AB_intra_dict[key]:
		AB_inlink_dict[out_link].append(key)
	print i,temp
	i = i + 1

print "Done"

for key in AB_inlink_dict:
	AB_inlink_dict[key] = list(set(AB_inlink_dict[key]))

print "Done"

save_pickle(AB_inlink_dict,"../../AB_inlink_dict")


