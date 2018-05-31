import os,sys,pickle,re

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
	file_name = file_name.replace(" ","_")
	if file_name[-5:] != ".wiki":
		file_name = file_name + ".wiki"
	try:
		file_lines_fp = open(path+file_name,"r")
	except IOError:
		print file_name
		return []
	file_data = file_lines_fp.read()	
	wikilinks = re.findall(r'\[\[[^ ]{1,}\|\|',file_data)	
	wikilinks = [ getWikiFileFormat(i[2:-2].split('#',1)[0]) for i in wikilinks ] 	
	return wikilinks



AB_intra_dict = load_pickle("../pickle_files/AB_intra_dict")

AB_inlink_dict = dict()

all_keys_inlink_dict = list()

for key in AB_intra_dict:
	all_keys_inlink_dict.extend(AB_intra_dict[key])
all_keys_inlink_dict = list(set(all_keys_inlink_dict))

print "Done"
all_keys_inlink_dict = [ getWikiFileFormat(i) for i in all_keys_inlink_dict] 
for key in all_keys_inlink_dict:
	AB_inlink_dict[key] = list()

print "Done"

print len(all_keys_inlink_dict)
temp = len(AB_intra_dict)

i = 0
for key in AB_intra_dict:
	for out_link in AB_intra_dict[key]:
		out_link = getWikiFileFormat(out_link)
		AB_inlink_dict[out_link].append(key)
	print i,temp
	i = i + 1

print "Done"

for key in AB_inlink_dict:
	AB_inlink_dict[key] = list(set(AB_inlink_dict[key]))

print "Done"

save_pickle(AB_inlink_dict,"../pickle_files/AB_inlink_dict")


