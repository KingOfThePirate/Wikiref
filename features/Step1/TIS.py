import os,sys,pickle,gensim,re
from nltk.tokenize import word_tokenize
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

def getSimilarity(line1,line2):
	line1 = re.sub(r"\[[\d]{1,3}\]"," ",line1)
	line2 = re.sub(r"\[[\d]{1,3}\]"," ",line2)
	line1 = line1.decode('latin-1')
	line2 = line2.decode('latin-1')
	query_doc1 = [w.lower() for w in word_tokenize(line1)]
	query_doc_bow1 = dictionary.doc2bow(query_doc1)
	query_doc_tf_idf1 = tf_idf[query_doc_bow1]

	query_doc2 = [w.lower() for w in word_tokenize(line2)]
	query_doc_bow2 = dictionary.doc2bow(query_doc2)
	query_doc_tf_idf2 = tf_idf[query_doc_bow2]

	index = gensim.similarities.MatrixSimilarity([query_doc_tf_idf1],num_features=len(dictionary))
	simi = index[query_doc_tf_idf2]
	return round(simi,3)

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
tfidf_summary_feature = open("tfidf_summary_feature","w")

dictionary = gensim.corpora.Dictionary.load("../../pickle_files/dictionary")
tf_idf = gensim.models.TfidfModel.load("../../pickle_files/tf_idf")

AB_summary_dict = dict()

count = 1
for AB in AB_overlap_relation:
	A,Bs,YN = parseAB(AB)
	Bs,YN = getTopBs(A,Bs,YN)
	yes = list()
	no = list()
	tfidf_summary_feature.write("@@"+A+"@@\n")
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
		tfidf_simi = getSimilarity(A_sum,B_sum)

		if YN[x] == "Y":
			tfidf_summary_feature.write(B+"##Y##"+str(tfidf_simi)+"\n")
			yes.append(tfidf_simi)
		else:
			tfidf_summary_feature.write(B+"##N##"+str(tfidf_simi)+"\n")
			no.append(tfidf_simi)
	yes.sort()
	no.sort()
	if len(yes) != 0:
		tfidf_summary_feature.write("$$$$Result_Y$$$$\n")
		tfidf_summary_feature.write("Min:"+str(yes[0])+"\n")
		tfidf_summary_feature.write("Max:"+str(yes[-1])+"\n")
		tfidf_summary_feature.write("Median:"+str(yes[len(yes)/2])+"\n")
		tfidf_summary_feature.write("Mean:"+str(round(sum(yes)/float(len(yes)),3))+"\n")
	if len(no) != 0:
		tfidf_summary_feature.write("$$$$Result_N$$$$\n")
		tfidf_summary_feature.write("Min:"+str(no[0])+"\n")
		tfidf_summary_feature.write("Max:"+str(no[-1])+"\n")
		tfidf_summary_feature.write("Median:"+str(no[len(no)/2])+"\n")
		tfidf_summary_feature.write("Mean:"+str(round(sum(no)/float(len(no)),3))+"\n")
	print "Done",A,len(AB_overlap_relation)-count
	count = count + 1
tfidf_summary_feature.close()
save_pickle(AB_summary_dict,"AB_summary_dict")