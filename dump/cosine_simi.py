import gensim,os,sys,pickle
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


relation_between_AB = open("3842_new_out_full","r").read().split("@@New@@\n")[1:]
cosine_simi_analysis = open("cosine_simi_analysis_"+sys.argv[1],"a")
wiki_vector_ConVec_Heuristic = load_pickle("wiki_vector_ConVec_Heuristic")
wiki_id_to_wiki_name = load_pickle("wiki_id_to_wiki_name")
wiki_name_to_wiki_id = load_pickle("wiki_name_to_wiki_id")


parts = int(sys.argv[1])

len_by_8 = len(relation_between_AB)/8
start_relation_AB = [ i*len_by_8 for i in range(8)]
end_relation_AB = [ i*len_by_8 for i in range(1,9)]
end_relation_AB[-1] = len(relation_between_AB)

for each_AB in relation_between_AB[start_relation_AB[parts]:end_relation_AB[parts]]:
	A,Bs,YN = parseAB(each_AB)
	if A is None:
		continue
	try:
		A_vec = wiki_vector_ConVec_Heuristic[wiki_name_to_wiki_id[A]]
	except KeyError:
		print "*******",A
		continue
	CS_YES = list()
	CS_NO = list()
	cosine_simi_analysis.write("@@"+parse[0]+"@@\n")
	for i in range(len(Bs)):
		try:
			B_vec = wiki_vector_ConVec_Heuristic[wiki_name_to_wiki_id[Bs[i]]]
		except KeyError:
			print "########",Bs[i]
			continue
		cosine_simi = round(cosine_similarity([A_vec],[B_vec])[0][0],4)
		if YN[i] == "YES":
			CS_YES.append(cosine_simi)
			cosine_simi_analysis.write(Bs[i]+"##Y##"+str(cosine_simi)+"\n")
		else:
			CS_NO.append(cosine_simi)
			cosine_simi_analysis.write(Bs[i]+"##N##"+str(cosine_simi)+"\n")
	CS_YES.sort()
	CS_NO.sort()
	if len(CS_YES) == 0:
		cosine_simi_analysis.write("$$$$Result_Y$$$$\n")
		cosine_simi_analysis.write("Min:1\n")
		cosine_simi_analysis.write("Max:1\n")
		cosine_simi_analysis.write("Median:1\n")
		cosine_simi_analysis.write("Mean:1\n")
	else:
		cosine_simi_analysis.write("$$$$Result_Y$$$$\n")
		cosine_simi_analysis.write("Min:"+str(CS_YES[0])+"\n")
		cosine_simi_analysis.write("Max:"+str(CS_YES[-1])+"\n")
		cosine_simi_analysis.write("Median:"+str(CS_YES[len(CS_YES)/2])+"\n")
		cosine_simi_analysis.write("Mean:"+str(round(sum(CS_YES)/len(CS_YES)*1.0,3))+"\n")
	if len(CS_NO) == 0:
		cosine_simi_analysis.write("$$$$Result_N$$$$\n")
		cosine_simi_analysis.write("Min:0\n")
		cosine_simi_analysis.write("Max:0\n")
		cosine_simi_analysis.write("Median:0\n")
		cosine_simi_analysis.write("Mean:0\n")
	else:
		cosine_simi_analysis.write("$$$$Result_Y$$$$\n")
		cosine_simi_analysis.write("Min:"+str(median_y1[0])+"\n")
		cosine_simi_analysis.write("Max:"+str(median_y1[-1])+"\n")
		cosine_simi_analysis.write("Median:"+str(median_y1[median_y_len1/2])+"\n")
		cosine_simi_analysis.write("Mean:"+str(round(sum(median_y1)/float(median_y_len1),3))+"\n")