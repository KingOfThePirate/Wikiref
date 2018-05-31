import gensim,os,sys,re,jellyfish
from nltk.tokenize import word_tokenize
import bisect

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

	a_file = getFileinLine(A,"../crawler_cs/")
	if a_file is None:
		return None,None,None,None
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
		return p+c,c,c+n,p+c+n
	return None,None,None,None
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
	a_file = getFileinLine(B,"../crawler_cs/")
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
def getRef(name,parent="../crawler_cs/"):
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


dictionary = gensim.corpora.Dictionary.load("dictionary")
tf_idf = gensim.models.TfidfModel.load("tf_idf")

all_A = open("3842","r")
parts = int(sys.argv[1])
relation_between_AB = open("3842_new_out_full","r").read().split("@@New@@\n")[1:]
pc_cite_simi_analysis = open("cite_simi_analysis_pc_"+sys.argv[1],"a")
c_cite_simi_analysis = open("cite_simi_analysis_c_"+sys.argv[1],"a")
cn_cite_simi_analysis = open("cite_simi_analysis_cn_"+sys.argv[1],"a")
pcn_cite_simi_analysis = open("cite_simi_analysis_pcn_"+sys.argv[1],"a")

len_by_8 = len(relation_between_AB)/8
start_relation_AB = [ i*len_by_8 for i in range(8)]
end_relation_AB = [ i*len_by_8 for i in range(1,9)]
end_relation_AB[-1] = len(relation_between_AB)

for each_AB in relation_between_AB[start_relation_AB[parts]:end_relation_AB[parts]]:
	A,Bs = parseAB(each_AB)
	if A is None:
		continue
	ref_dict_A = getRef(A)
	if ref_dict_A is None:
			continue
	pc_cite_simi_analysis.write("@@NewA@@\nA:"+A+"\n")
	c_cite_simi_analysis.write("@@NewA@@\nA:"+A+"\n")
	cn_cite_simi_analysis.write("@@NewA@@\nA:"+A+"\n")
	pcn_cite_simi_analysis.write("@@NewA@@\nA:"+A+"\n")
	for B in Bs:
		pc,c,cn,pcn = getCiteContextofBinA(A,B)
		if pc is None:
			continue

		ref_dict_B = getRef(B)
		if ref_dict_B is None:
			continue

		ref_context = getRefContext(B,ref_dict_B)
		if ref_context is None:
			continue

		similar_ref_AB = getSimilarRefAB(ref_dict_A,ref_dict_B)

		pc_cite_simi_analysis.write("@@NewB@@\nB:"+B+"\n")
		c_cite_simi_analysis.write("@@NewB@@\nB:"+B+"\n")
		cn_cite_simi_analysis.write("@@NewB@@\nB:"+B+"\n")
		pcn_cite_simi_analysis.write("@@NewB@@\nB:"+B+"\n")

		for key in ref_context:
			simi_pc = getSimilarity(ref_context[key],pc)
			simi_c = getSimilarity(ref_context[key],c)
			simi_cn = getSimilarity(ref_context[key],cn)
			simi_pcn = getSimilarity(ref_context[key],pcn)
			if key in similar_ref_AB:
				pc_cite_simi_analysis.write(str(key)+"$$"+str(simi_pc)+"$$"+"YES\n")
				c_cite_simi_analysis.write(str(key)+"$$"+str(simi_c)+"$$"+"YES\n")
				cn_cite_simi_analysis.write(str(key)+"$$"+str(simi_cn)+"$$"+"YES\n")
				pcn_cite_simi_analysis.write(str(key)+"$$"+str(simi_pcn)+"$$"+"YES\n")
			else:
				pc_cite_simi_analysis.write(str(key)+"$$"+str(simi_pc)+"$$"+"NO\n")
				c_cite_simi_analysis.write(str(key)+"$$"+str(simi_c)+"$$"+"NO\n")
				cn_cite_simi_analysis.write(str(key)+"$$"+str(simi_cn)+"$$"+"NO\n")
				pcn_cite_simi_analysis.write(str(key)+"$$"+str(simi_pcn)+"$$"+"NO\n")


pc_cite_simi_analysis.close()
c_cite_simi_analysis.close()
cn_cite_simi_analysis.close()
pcn_cite_simi_analysis.close()

"""
pc_cite_simi_analysis.write()
c_cite_simi_analysis.write()
cn_cite_simi_analysis.write()
pcn_cite_simi_analysis.write()
"""