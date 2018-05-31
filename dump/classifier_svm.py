from sklearn import svm
import csv,math,sys,pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import f1_score
from random import shuffle
import numpy as np
import matplotlib.pyplot as plt
import gensim,sys,re,jellyfish
from nltk.tokenize import word_tokenize
import bisect

def save_pickle(object_to_save,filename):
	fileObject = open(filename,'wb')
	pickle.dump(object_to_save,fileObject) 
	fileObject.close()

def load_pickle(filename):
	fileObject = open(filename,'rb') 
	b = pickle.load(fileObject) 
	fileObject.close()
	return b

def getWikilinksNo():
	filename = "count_3_3.csv"
	with open(filename, 'r') as csvfile:
		# creating a csv reader object
		csvreader = csv.reader(csvfile, delimiter = ';')
		# extracting field names through first row
		fields = csvreader.next()
		pre_reca_dict = dict()
		# extracting each data row one by one
		for row in csvreader:
			if "TLDR.wiki" in row:
				pre_reca_dict['TL;DR.wiki'] = 71
			else:
				pre_reca_dict[row[0]] = int(row[5])
	return pre_reca_dict

def removeBs(a):
	ret = list()
	bs = list()
	for i in a:
		ret.append(i[1:])
		bs.append(i[0])
	return ret

def removeFeatures(a):
	ret = list()
	bs = list()
	for i in a:
		ret.append(i[1:])
		bs.append(i[0])
	return bs

def getTrainingTestingData(filename,train_size,scale = 1):
	training_size = int(math.ceil(train_size*(1.0-training_testing_variation)))
	# print training_size
	file_p = open(filename,"r").read().splitlines()
	wiki_dict = dict()
	key = ""
	for line in file_p:
		if line[-5:] == ".wiki":
			key = line
			wiki_dict[key] = {'Y':[],'N':[]}
		else:
			tokens = line.split("#")
			wiki_dict[key][tokens[-1]].append(tokens[0:1]+[float(i) for i in tokens[1:-1]])
	keys = list(wiki_dict.keys())
	# Random-ness
	# shuffle(keys)
	# in_degree#out_degree#tfidf#in_sent#out_sent#class
	train_yes = list()
	train_no = list()
	for key in keys:
		if len(train_yes) < training_size and len(train_no) < training_size:
			train_yes = train_yes + removeBs(wiki_dict[key]['Y'])
			train_no = train_no + removeBs(wiki_dict[key]['N'][0:len(wiki_dict[key]['Y'])])
			del wiki_dict[key]
		elif len(train_yes) < training_size :
			train_yes = train_yes + removeBs(wiki_dict[key]['Y'])
			del wiki_dict[key]
		else:
			break
	# Random-ness
	# shuffle(train_no)
	# train_no = train_no[0:training_size]
	return wiki_dict,train_yes,train_no

def getXY(file_name):
	X = list()
	Y = list()
	csvfile = open(file_name, 'r').read().splitlines()
	# Randomness
	# shuffle(csvfile)
	for row in csvfile:
		row = row.split("#")
		one_x = list()
		for x1 in row[0:-1]:
			one_x.append(float(x1))
		X.append(one_x)
		Y.append(row[-1])
	return [X,Y]

def my_train_test_split(X_no,y_no,train_size,scale = 1):
	training_size = int(math.ceil(train_size*(1.0-training_testing_variation)))
	testing_size = train_size - training_size
	testing_size = testing_size*scale
	return [ X_no[0:training_size] , X_no[training_size:training_size+testing_size] , y_no[0:training_size] , y_no[training_size:training_size+testing_size] ]

def clean_cb(text):
	clean_text = text
	for x in xrange(len(text)-1,-1,-1):
		if text[x] == ']':
			clean_text = text[:x]
	return clean_text

def clean(line):
	line_len = len(line)
	if line_len == 0 or line is None:
		return ""
	new_line = ""
	in_loop = False
	balancing = 0
	for x in range(0,line_len):
		if line[x:x+2] == "[[" or in_loop :
			in_loop = True
			if line[x] == "[":
				balancing = balancing + 1
			elif line[x] == "]":
				balancing = balancing - 1

			if line[x:x+2] == "[[":
				open_b = x
			elif line[x:x+2] == "||":
				bar = x
			elif line[x:x+2] == "]]":
				close_b = x
			if balancing == 0:
				try:
					in_loop = False
					link = line[open_b+2:bar]
					link_text = clean_cb(line[bar+2:close_b])
					new_line = new_line + " " + link_text
					balancing = 0
				except UnboundLocalError:
					print "error clean",line
					return re.sub(r"[^A-Za-z]",	" ",line)
		else:
			new_line = new_line + line[x]
	return new_line

def getFileinLine(file_name,path):
	try:
		file_wiki = open(path+file_name,"r").read().splitlines()
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
		# lines  = file_wiki_i.split(". ")
		lines = list()
		start = 0
		for m in re.finditer(r'\.\[\d{1,3}\]\s[A-Z]|\.\s[A-Z]',file_wiki_i):
			lines.append(file_wiki_i[start:m.end(0)-1]) 
			start = m.end(0)-1
		lines.append(file_wiki_i[start:])
		for lines_i in lines:
			if len(lines_i) > 5 and lines_i[0:2] != "##":
				file_lines.append(lines_i)
	if len(file_lines) == 0:
		return None

	file_lines_return = list()
	curr_line = file_lines[0]
	for line_i in range(1,len(file_lines)):
		if file_lines[line_i][0].isupper():
			file_lines_return.append(curr_line)
			curr_line = file_lines[line_i]
		else:
			curr_line = curr_line + file_lines[line_i]
	file_lines_return.append(curr_line)
	return file_lines_return

def parseAB(text):
	if len(text)==0 or text is None:
		return None,[]
	lines = text.splitlines()
	A = lines[0][2:].split("##__##")[0]
	only_yes_B = list()
	for line in lines:
		if "YES" == line.rsplit(":",1)[-1]:
			only_yes_B.append(line.split("##__##")[1].rsplit(":",1)[0])
	return A,only_yes_B

def getCiteContextofBinA(A,B_i):
	B_i = B_i.replace(" ","_")
	if B_i[-5:] == ".wiki":
		B_i = B_i[0:-5]
	A = A.replace(" ","_")
	if A[-5:] != ".wiki":
		A = A + ".wiki"

	a_file = getFileinLine(A,"../similarity/crawler_cs/")
	if a_file is None:
		return None
	p,c,n = "","",""
	for a_i in range(len(a_file)):
		index_of_b = a_file[a_i].find("[["+B_i+"||")
		if index_of_b == -1:
			continue
		if a_i > 0:
			p = clean(a_file[a_i-1])
		c = clean(a_file[a_i])
		if a_i < len(a_file)-1:
			n = clean(a_file[a_i+1])
		return p+c+n
	return None

def find_ref(cite_numbers,line):
	found = list()
	for no in cite_numbers:
		if no in line:
			found.append(no)
	return list(set(found))

def getRefContext(B,ref_dict):
	B = B.replace(" ","_")
	if B[-5:] != ".wiki":
		B = B+".wiki"
	a_file = getFileinLine(B,"../similarity/crawler_cs/")
	if a_file is None:
		return None
	no_ref = len(ref_dict)
	cite_numbers = ["["+str(i)+"]" for i in range(1,no_ref+1)]
	
	ref_context = dict()
	for i in range(1,no_ref+1):
		ref_context[i] = ""

	for file_i in xrange(len(a_file)):
		found = find_ref(cite_numbers,a_file[file_i])
		found = [int(i[1:-1]) for i in found]
		for i in found:
			ref_context[i] = ref_context[i] + " " + a_file[file_i]
	for i in ref_context:
		ref_context[i] = clean(ref_context[i])
	return ref_context

def getRef(name,parent="../similarity/crawler_cs/"):
	try:
		curr = open(parent+name,"r")
	except IOError:
		print "getRef",name
		return None
	curr_lines = curr.read().splitlines()
	curr.close()
	try :
		ref_index = curr_lines.index("##References") + 1
		len_curr_lines = len(curr_lines)
		list_of_ref_curr = dict()
		count = 0;
		for j in xrange(ref_index,len_curr_lines,1):
			curr_line_split = curr_lines[j].split(":",1)
			if len(curr_line_split) <= 1:
				continue
			curr_line_split[0] = curr_line_split[0].strip(' ')
			curr_line_split[1] = curr_line_split[1].strip(' ')
			if "Year" == curr_line_split[0]:
				year = curr_line_split[1].split("$$")[0]
				number = int(curr_line_split[1].split("$$")[1])
				count = count + 1
			elif "Authors" == curr_line_split[0]:
				author = curr_line_split[1]
				count = count + 1
			elif "Reference" == curr_line_split[0]:
				ref = curr_line_split[1]
				if count == 2 :
					list_of_ref_curr[number] = {'year':year,'authors':author,'ref':ref}
					count = 0
		return list_of_ref_curr
	except Exception:
		print "getRef",name
		print sys.exc_info()[0]
		return None

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
	return round(simi,2)

def getSimilarRefAB(a,b):
	similar_ref = list()
	for key_a in a:
		for key_b in b:
			a_au = unicode(a[key_a]['authors'],"utf-8")
			b_au = unicode(b[key_b]['authors'],"utf-8")
			a_ref = unicode(a[key_a]['ref'],"utf-8")
			b_ref = unicode(b[key_b]['ref'],"utf-8")
			jaro_au = jellyfish.jaro_winkler(a_au, b_au)
			jaro_ref = jellyfish.jaro_winkler(a_ref, b_ref)
			if (jaro_au >= 0.70 and len(a_au)>5 and len(b_au)>5) and (jaro_ref >= 0.75 and len(a_ref)>5 and len(b_ref)>5):
				similar_ref.append(key_b)
	similar_ref = list(set(similar_ref))
	return similar_ref

def whichSimilarRefAB(a,b,index_b):

	for key_a in a:
		a_au = unicode(a[key_a]['authors'],"utf-8")
		b_au = unicode(b[index_b]['authors'],"utf-8")
		a_ref = unicode(a[key_a]['ref'],"utf-8")
		b_ref = unicode(b[index_b]['ref'],"utf-8")
		jaro_au = jellyfish.jaro_winkler(a_au, b_au)
		jaro_ref = jellyfish.jaro_winkler(a_ref, b_ref)
		if (jaro_au >= 0.70 and len(a_au)>5 and len(b_au)>5) and (jaro_ref >= 0.75 and len(a_ref)>5 and len(b_ref)>5):
			return key_a
	return None

def isSimilarRefAB(a,b):
	a_au = unicode(a['authors'],"utf-8")
	b_au = unicode(b['authors'],"utf-8")
	a_ref = unicode(a['ref'],"utf-8")
	b_ref = unicode(b['ref'],"utf-8")
	jaro_au = jellyfish.jaro_winkler(a_au, b_au)
	jaro_ref = jellyfish.jaro_winkler(a_ref, b_ref)
	if (jaro_au >= 0.70 and len(a_au)>5 and len(b_au)>5) and (jaro_ref >= 0.75 and len(a_ref)>5 and len(b_ref)>5):
		return True
	return False

def getUniqueRef(ref_list):
	uniq_list = list()
	for a in ref_list:
		flag = True
		for b in uniq_list:
			if isSimilarRefAB(a,b):
				flag = False
				break
		if flag:
			uniq_list.append(a)
	return uniq_list

def merge(a,b,a_id,b_id):
	a = a[::-1]
	b = b[::-1]
	a_id = a_id[::-1]
	b_id = b_id[::-1]
	i = 0
	j = 0
	c = list()
	while i < len(a) and j < len(b):
		if a[i] < b[j]:
			c.append(b_id[j])
			j = j + 1
		else:
			c.append(a_id[i])
			i = i + 1
	while i < len(a):
		c.append(a_id[i])
		i = i + 1
	while j < len(b):
		c.append(b_id[j])
		j = j + 1
	return c

def getPrecisionAtK(a,k = 3):
	count  = 0
	k = min(len(a),k)
	for i in range(k):
		if a[i] == "YES":
			count += 1
	return round((count)/(k*1.0),2)

def getRecallAtK(a,number_of_yes,k = 3):
	count  = 0
	k = min(len(a),k)
	for i in range(k):
		if a[i] == "YES":
			count += 1
	return round((count)/(number_of_yes*1.0),2)

def getBsRef(A,Bs):
	ref_dict_A = getRef(A)
	if ref_dict_A is None:
			return None
	ref_overlap_A = set()
	ref_by_b = list()
	for B in Bs:
		pcn = getCiteContextofBinA(A,B)
		if pcn is None:
			continue

		ref_dict_B = getRef(B)
		if ref_dict_B is None:
			continue

		ref_context = getRefContext(B,ref_dict_B)
		if ref_context is None:
			continue
		similar_ref_AB = getSimilarRefAB(ref_dict_A,ref_dict_B)

		ref_yes = list()
		ref_no = list()
		ref_yes_id = list()
		ref_no_id = list()

		for key in ref_context:
			simi_pcn = getSimilarity(ref_context[key],pcn)
			if key in similar_ref_AB:
				ref_yes.append(simi_pcn)
				ref_yes_id.append(key)
			else:
				ref_no.append(simi_pcn)
				ref_no_id.append(key)
		c = merge(ref_yes,ref_no,ref_yes_id,ref_no_id)
		try:
			for i in range(min(3,len(c))):
				temp = whichSimilarRefAB(ref_dict_A,ref_dict_B,c[i])
				ref_by_b.append(ref_dict_B[c[i]])
				if temp is None:
					continue
				ref_overlap_A.add(temp)
		except IndexError:
			pass
	return ref_by_b,ref_overlap_A,ref_dict_A


# 14608(0.7,yes) 14895(0.047,no)
dictionary = gensim.corpora.Dictionary.load("dictionary")
tf_idf = gensim.models.TfidfModel.load("tf_idf")
X_no,y_no = getXY("features_no.csv")
X_yes,y_yes = getXY("features_yes.csv")

training_testing_variation = 0.3

wiki_As,train_yes,train_no = getTrainingTestingData("features4.csv",len(X_yes),scale = 1)

print "No. yes",len(train_yes)
print "No. no",len(train_no)
print "Test set of A's",len(wiki_As)
X_train = train_yes + train_no
y_train = ['Y']*len(train_yes) + ['N']*len(train_no)
# Linear Kernel
svc_linear = svm.SVC(kernel='linear', C = 1.0)
svc_linear.fit(X_train, y_train)
print "Number of training ",len(X_train)
# file_w = open("for_each_A","w")

keys = list(wiki_As.keys())
pre_reca_dict = dict()
no_wikilinks = getWikilinksNo()
mymax = 0
pre = list()
rec = list()

by_8 = (len(keys)+8)/8
part = int(sys.argv[1])

for key in keys[part*by_8:(part+1)*by_8]:
	print "##New##"
	print key
	if len(wiki_As[key]['Y']) == 0 or len(wiki_As[key]['N']) == 0:
		continue
	number_yes = len(wiki_As[key]['Y'])
	number_no = len(wiki_As[key]['N'])
	X_test = removeBs(wiki_As[key]['Y']) + removeBs(wiki_As[key]['N'])
	X_test_Bs = removeFeatures(wiki_As[key]['Y']) + removeFeatures(wiki_As[key]['N'])
	y_test = ['Y']*number_yes + ['N']*number_no
	predicted = svc_linear.predict(X_test)
	# print X_test_Bs
	# print y_test
	# print predicted
	predicted_yes = [X_test_Bs[i] for i in range(len(predicted)) if predicted[i] == 'Y' ]
	# print len(predicted),len(y_test)
	cnf_matrix = confusion_matrix(y_test, predicted)
	report = classification_report(y_test, predicted)
	y_pred = predicted
	As_ref = getRef(key)
	if As_ref is None:
		continue
	ref_by_b,ref_overlap_A,ref_dict_A = getBsRef(key,predicted_yes)
	ref_by_b = getUniqueRef(ref_by_b)
	
	if len(ref_by_b) < 1:
		pre.append(1.0)
	else:	
		pre.append(len(ref_overlap_A)*1.0/len(ref_by_b))

	if len(ref_dict_A) < 1:
		rec.append(1.0)
	else:
		rec.append(len(ref_overlap_A)*1.0/len(ref_dict_A))

print "Average Precision : ",round(sum(pre)/len(pre),2)
print "Average Recall : ",round(sum(rec)/len(rec),2)

save_pickle(pre,"pre_"+sys.argv[1])
save_pickle(rec,"rec_"+sys.argv[1])