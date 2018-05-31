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

def getWikiFileFormat(name):
	name = name.replace(" ","_")
	name = name.split("#",1)[0]
	name = name.replace("/","$$$")
	if len(name)<5 or name[-5:] != ".wiki":
		name = name + ".wiki"
	return name

def getIntraLinks_OnlyTop(file_name,path):
	file_lines_fp = open(path+file_name,"r")
	file_lines = file_lines_fp.read().splitlines()
	file_lines_fp.close()
	intra = list()
	s = 2
	try:
		e1 = file_lines.index("See also[edit]")
	except ValueError:
		e1 = len(file_lines)
	try:
		e2 = file_lines.index("References[edit]")
	except ValueError:
		e2 = len(file_lines)
	try:
		e3 = file_lines.index("Further reading[edit]")
	except ValueError:
		e3 = len(file_lines)
	try:
		e4 = file_lines.index("External links[edit]")
	except ValueError:
		e4 = len(file_lines)
	e = min(e1,e2,e3,e4,len(file_lines))

	for line in file_lines[s:e]:
		line_intra = re.findall(r'\[\[[A-Za-z0-9_\%\#]{1,}\|\|',line)
		line_intra = [ getWikiFileFormat(i[2:-2]) for i in line_intra ]
		intra.extend(line_intra)
	return list(set(intra))

ALL_As_Bs = load_pickle("../../ALL_As_Bs")
AB_intra_dict = dict()

print len(ALL_As_Bs)

for AB in ALL_As_Bs:
	AB_intra_dict[AB] = getIntraLinks_OnlyTop(AB,"../../../crawler_phy/")

save_pickle(AB_intra_dict,"../../AB_intra_dict")