import pickle,os,sys


def save_pickle(object_to_save,filename):
	fileObject = open(filename,'wb')
	pickle.dump(object_to_save,fileObject) 
	fileObject.close()

def load_pickle(filename):
	fileObject = open(filename,'rb') 
	b = pickle.load(fileObject) 
	fileObject.close()
	return b

def get_depth_lca(a,b):
	count = 0
	for i in range(min(len(a),len(b))):
		if a[i] == b[i]:
			count += 1
		else:
			break
	return count

def getSimilarityCategory(cat_a,cat_b):
	try:
		path_a = paths_from_root_to_all_nodes[cat_a]
		path_b = paths_from_root_to_all_nodes[cat_b]
	except KeyError:
		return 0
	if len(path_a)==0 or len(path_b)==0:
		return 0
	depth_lca_max = 0.0
	a_max = 0.0
	b_max = 0.0
	for a in path_a:
		for b in path_b:
			depth_lca = get_depth_lca(a,b)
			if depth_lca > depth_lca_max:
				depth_lca_max = depth_lca
				a_max = a
				b_max = b
	return (2.0*depth_lca_max)/(len(a_max)+len(b_max))

def getSimilarityCategories(cats_a,cats_b):
	category_list = list()
	for a in cats_a:
		for b in cats_b:
			temp = getSimilarityCategory(a,b)
			category_list.append(temp)
	if len(category_list) == 0:
		return 0.0
	return round(sum(category_list)/(1.0*len(category_list)),2)

# def getSimilarityCategories(cats_a,cats_b):
# 	category_list = list()
# 	for a in cats_a:
# 		max_b = 0.0
# 		for b in cats_b:
# 			temp = getSimilarityCategory(a,b)
# 			max_b = max(max_b,temp)
# 		category_list.append(max_b)
# 	if len(category_list) == 0:
# 		return 0.0
# 	return round(sum(category_list)/(1.0*len(category_list)),2)

def parseAB(text):
	if len(text)==0 or text is None:
		return None,[]
	lines = text.strip().splitlines()
	A = lines[0][2:].split("##__##")[0]
	Bs = list()
	YN = list()
	for line in lines:
		Bs.append(line.split("##__##")[1].rsplit(":",1)[0])
		YN.append(line.split("##__##")[1].rsplit(":",1)[1])
	return A,Bs,YN

def getCategory(file_name):
	try:
		file_p = open("../crawler_cs/"+file_name,"r")
	except IOError:
		print "IOError ",file_name
		return None
	
	file_lines = file_p.read().splitlines()
	file_p.close()
	try:
		index_cat = file_lines.index("##Categories:")
		index_intra = file_lines.index("##IntraLinks:")
	except ValueError:
		return None
	return file_lines[index_cat+1:index_intra]

paths_from_root_to_all_nodes = load_pickle("paths_from_root_to_all_nodes")
part = int(sys.argv[1])
CN_simi_analysis = open("CN_simi_analysis_meanofmean_"+str(part),"w")
new_out = open("3842_new_out_full","r").read().split("@@New@@")[1:]
by8 = (len(new_out)+8)/8

for each_A in new_out[part*by8:(part+1)*by8]:
	A,Bs,YN = parseAB(each_A)
	if A is None:
		continue
	categories_A = getCategory(A)
	if categories_A is None:
		continue
	yes = list()
	no = list()
	CN_simi_analysis.write("@@"+A+"@@\n")
	for B_i in range(len(Bs)):
		categories_B = getCategory(Bs[B_i])
		if categories_B is None:
			continue
		cat_simi = getSimilarityCategories(categories_A,categories_B)
		if YN[B_i] == "YES":
			CN_simi_analysis.write(Bs[B_i]+"##Y##"+str(cat_simi)+"\n")
			yes.append(cat_simi)
		elif YN[B_i] == "NO":
			CN_simi_analysis.write(Bs[B_i]+"##N##"+str(cat_simi)+"\n")
			no.append(cat_simi)
	yes.sort()
	no.sort()
	if len(no) == 0:
		no.append(0)
	if len(yes) == 0:
		yes.append(0)
	CN_simi_analysis.write("$$$$Result_Y$$$$\n")
	CN_simi_analysis.write("Min:"+str(yes[0])+"\n")
	CN_simi_analysis.write("Max:"+str(yes[-1])+"\n")
	CN_simi_analysis.write("Median:"+str(yes[len(yes)/2])+"\n")
	CN_simi_analysis.write("Mean:"+str(round(sum(yes)/len(yes)*1.0,3))+"\n")
	CN_simi_analysis.write("$$$$Result_N$$$$\n")
	CN_simi_analysis.write("Min:"+str(no[0])+"\n")
	CN_simi_analysis.write("Max:"+str(no[-1])+"\n")
	CN_simi_analysis.write("Median:"+str(no[len(no)/2])+"\n")
	CN_simi_analysis.write("Mean:"+str(round(sum(no)/len(no)*1.0,3))+"\n")

CN_simi_analysis.close()