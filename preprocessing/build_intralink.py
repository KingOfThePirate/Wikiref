import os,sys,re,pickle

def getBs(file_name,path):
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

def getFileData(filename,parent="./"):
	fp = open(parent+filename,"r")
	data = fp.read()
	fp.close()
	return data

def getWikiFileFormat(name):
	name = name.replace(" ","_")
	name = name.split("#",1)[0]
	name = name.replace("/","$$$")
	if len(name)<5 or name[-5:] != ".wiki":
		name = name + ".wiki"
	return name

def save_pickle(object_to_save,filename):
	fileObject = open(filename,'wb')
	pickle.dump(object_to_save,fileObject) 
	fileObject.close()

path = "../dataset/ComputerScience/" # Path for dataset
list_of_As = getFileData("../dataset/ComputerScience_As").splitlines() # FIle tha contains all the As

AB_intra_dict = dict()

for A in list_of_As:
	A = getWikiFileFormat(A)
	AB_intra_dict[A] = getBs(A,path)

save_pickle(AB_intra_dict,"../pickle_files/AB_intra_dict")