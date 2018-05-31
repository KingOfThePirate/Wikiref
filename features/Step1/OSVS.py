import os,sys,pickle,re,gensim,torch,nltk
from sklearn.metrics import jaccard_similarity_score
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity

def save_pickle(object_to_save,filename):
	fileObject = open(filename,'wb')
	pickle.dump(object_to_save,fileObject) 
	fileObject.close()

def load_pickle(filename):
	fileObject = open(filename,'rb') 
	b = pickle.load(fileObject) 
	fileObject.close()
	return b
def getFileData(filename,parent):
	fp = open(parent+filename,"r")
	data = fp.read()
	fp.close()
	return data
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

def getSimilarity(line1,line2):
	line1 = re.sub(r"\[[\d]{1,3}\]"," ",line1)
	line2 = re.sub(r"\[[\d]{1,3}\]"," ",line2)
	if len(line1) == 0:
		line1 = "empty"
	if len(line2) == 0:
		line2 = "empty"
	line1 = line1.decode('utf-8').strip()
	line2 = line2.decode('utf-8').strip()
	vec1,vec2 = getSent2Vec([line1,line2])
	return round(cosine_similarity([vec1],[vec2])[0][0],4)

def getSent2Vec(input_sentences):
	infersent = torch.load("encoder/infersent.allnli.pickle" , map_location=lambda storage, loc: storage)
	infersent.set_glove_path("GloVe/glove.840B.300d.txt")
	infersent.build_vocab(input_sentences, tokenize=True)
	summary_embeddings = infersent.encode(input_sentences, tokenize=True)
	return summary_embeddings


def getFileinLine(file_name,path):
	file_name = file_name.replace(" ","_")
	if file_name[-5:] != ".wiki":
		file_name = file_name + ".wiki"
	
	try:
		file_wiki = getFileData(file_name,path).splitlines()
	except IOError:
		print "getFileLine",file_name
		return None
	try:
		content_e = file_wiki.index("##Categories:")
	except ValueError:
		return None
	file_wiki = file_wiki[2:content_e]
	file_lines = list()
	for file_wiki_i in file_wiki:
		lines  = file_wiki_i.split(". ")
		for lines_i in lines:
			if len(lines_i) > 5 and lines_i[0:2] != "##" and lines_i[0:2] != "==":
				file_lines.append(lines_i)
	return file_lines

def getLine(file_name,target,path):
	lines = getFileinLine(file_name,path)
	if lines is None:
		return ""
	if target[-5:] == ".wiki":
		link = target[0:-5]
	else:
		link = target
	link = link.replace(" ","_")
	return_line = ""
	for i in xrange(0,len(lines)):
		if link in lines[i]:
			return_line_loop = lines[i]
			if i > 0:
				return_line_loop = lines[i-1] +". " +return_line_loop
			return_line = return_line + ". "  + return_line_loop
	return re.sub(r"\[[\d]{1,3}\]"," ",return_line)

def getTopBs(A,Bs,YN):
	Bs_ret = list()
	YN_ret = list()
	topBs = AB_intra_dict[A]
	for tb in topBs:
		try:
			index = Bs.index(tb)
		except ValueError:
			# print tb
			continue
		Bs_ret.append(Bs[index])
		YN_ret.append(YN[index])
	return Bs_ret,YN_ret

dataset_path = "../../dataset/ComputerScience/"
AB_overlap_relation = open("../../prepocessing/AB_overlap_relation","r").read().split("@@New@@\n")[1:]
AB_intra_dict = load_pickle("../../pickle_files/AB_intra_dict")
outsent_feature = open("outsentVec_feature","w")

dictionary = gensim.corpora.Dictionary.load("../../pickle_files/dictionary")
tf_idf = gensim.models.TfidfModel.load("../../pickle_files/tf_idf")


outsent_vec_data_dict = dict()
count = 0
for AB in AB_overlap_relation:
	A,Bs,YN = parseAB(AB)
	Bs,YN = getTopBs(A,Bs,YN)
	yes = list()
	no = list()
	outsent_feature.write("@@"+A+"@@\n")
	outsent_vec_data_dict[A] = dict()
	for x in range(len(Bs)):
		B = Bs[x]
		try:
			intersection = list(set(AB_intra_dict[A]) & set(AB_intra_dict[B]))
		except KeyError:
			print "KeyError",A,B
			continue
		A_outsent = ""
		B_outsent = ""
		A_outsent_vec = ""
		B_outsent_vec = ""
		outsent_simi = 0
		for intersection_i in intersection:
			A_outsent = getLine(A,intersection_i,dataset_path)
			B_outsent = getLine(B,intersection_i,dataset_path)
			A_outsent_vec = A_outsent_vec + " " + A_outsent
			B_outsent_vec = B_outsent_vec + " " + B_outsent
			outsent_simi = outsent_simi + getSimilarity(A_outsent,B_outsent)
		outsent_vec_data_dict[A][B] = dict()
		outsent_vec_data_dict[A][B][0] = A_outsent_vec
		outsent_vec_data_dict[A][B][1] = B_outsent_vec
		if len(intersection)==0:
			outsent_simi = 0.0
		else:
			outsent_simi = outsent_simi/(len(intersection)*1.0)
		if YN[x] == "Y":
			outsent_feature.write(B+"##Y##"+str(outsent_simi)+"\n")
			yes.append(outsent_simi)
		else:
			outsent_feature.write(B+"##N##"+str(outsent_simi)+"\n")
			no.append(outsent_simi)
	yes.sort()
	no.sort()

	if len(yes) != 0:
		outsent_feature.write("$$$$Result_Y$$$$\n")
		outsent_feature.write("Min:"+str(yes[0])+"\n")
		outsent_feature.write("Max:"+str(yes[-1])+"\n")
		outsent_feature.write("Median:"+str(yes[len(yes)/2])+"\n")
		outsent_feature.write("Mean:"+str(round(sum(yes)/float(len(yes)),3))+"\n")
	if len(no) != 0:
		outsent_feature.write("$$$$Result_N$$$$\n")
		outsent_feature.write("Min:"+str(no[0])+"\n")
		outsent_feature.write("Max:"+str(no[-1])+"\n")
		outsent_feature.write("Median:"+str(no[len(no)/2])+"\n")
		outsent_feature.write("Mean:"+str(round(sum(no)/float(len(no)),3))+"\n")
	count = count + 1
	print "Done",A,len(AB_overlap_relation)-count

save_pickle(outsent_vec_data_dict,"outsent_vec_data_dict")
outsent_feature.close()