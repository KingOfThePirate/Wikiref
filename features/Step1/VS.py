import os,sys,pickle,gensim,re,nltk
from nltk.tokenize import word_tokenize
from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics.pairwise import cosine_similarity
import torch

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

def getFileData(filename,parent):
	try:
		fp = open(parent+filename,"r")
	except IOError:
		print "IOError",filename
		return None
	data = fp.read()
	fp.close()
	return data

def getSummary(filename,parent=dataset_path):
	file_data  =getFileData(filename,parent)
	if file_data is None:
		return None
	file_data = file_data.splitlines()
	try:
		start_index = file_data.index("##Content:")
		cat_index = file_data.index("##Categories:")
		contente_index = file_data.index("Contents")
		edit_index = len(file_data)
		for line in file_data:
			if "[edit]" in line:
				edit_index = file_data.index(line)
				break
		end_index = min(cat_index,contente_index,edit_index)
	except ValueError:
		return ""
	ret_str = " ".join(file_data[start_index+1:end_index])
	ret_str = re.sub(r"[^A-Za-z0-9]",	" ",ret_str)
	return ret_str

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
sent2vec_summary_feature = open("summaryVec_feature","w")

dictionary = gensim.corpora.Dictionary.load("../../pickle_files/dictionary")
tf_idf = gensim.models.TfidfModel.load("../../pickle_files/tf_idf")

AB_summary_dict = dict()

count = 1
for AB in AB_overlap_relation:
	A,Bs,YN = parseAB(AB)
	Bs,YN = getTopBs(A,Bs,YN)
	yes = list()
	no = list()
	sent2vec_summary_feature.write("@@"+A+"@@\n")
	A_sum = getSummary(A)
	AB_summary_dict[A] = A_sum
	if A_sum is None:
		continue
	for x in range(len(Bs)):
		B = Bs[x]
		B_sum = getSummary(B)
		AB_summary_dict[B] = B_sum
		if B_sum is None:
			continue
		sent2vec_simi = getSimilarity(A_sum,B_sum)

		if YN[x] == "Y":
			sent2vec_summary_feature.write(B+"##Y##"+str(sent2vec_simi)+"\n")
			yes.append(sent2vec_simi)
		else:
			sent2vec_summary_feature.write(B+"##N##"+str(sent2vec_simi)+"\n")
			no.append(sent2vec_simi)
	yes.sort()
	no.sort()
	if len(yes) != 0:
		sent2vec_summary_feature.write("$$$$Result_Y$$$$\n")
		sent2vec_summary_feature.write("Min:"+str(yes[0])+"\n")
		sent2vec_summary_feature.write("Max:"+str(yes[-1])+"\n")
		sent2vec_summary_feature.write("Median:"+str(yes[len(yes)/2])+"\n")
		sent2vec_summary_feature.write("Mean:"+str(round(sum(yes)/float(len(yes)),3))+"\n")
	if len(no) != 0:
		sent2vec_summary_feature.write("$$$$Result_N$$$$\n")
		sent2vec_summary_feature.write("Min:"+str(no[0])+"\n")
		sent2vec_summary_feature.write("Max:"+str(no[-1])+"\n")
		sent2vec_summary_feature.write("Median:"+str(no[len(no)/2])+"\n")
		sent2vec_summary_feature.write("Mean:"+str(round(sum(no)/float(len(no)),3))+"\n")
	print "Done",A,len(AB_overlap_relation)-count
	count = count + 1
sent2vec_summary_feature.close()
save_pickle(AB_summary_dict,"AB_summary_dict")