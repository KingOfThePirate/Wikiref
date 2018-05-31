import gensim,os,sys,re
from nltk.tokenize import word_tokenize
import bisect,pickle,cPickle

def save_pickle(object_to_save,filename):
	fileObject = open(filename,'wb')
	pickle.dump(object_to_save,fileObject) 
	fileObject.close()

def load_pickle(filename):
	fileObject = open(filename,'rb') 
	b = pickle.load(fileObject) 
	fileObject.close()
	return b

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
	new_line = re.sub(r"[^A-Za-z0-9%]",	" ",new_line)
	return new_line
def getFileData(filename,parent = dataset_path):
	fp = open(parent+filename,"r")
	data = fp.read().splitlines()
	data = " ".join(data)
	fp.close()
	return data

dataset_path = "../dataset/ComputerScience/"

ALL_As_Bs = os.listdir(dataset_path)

raw_documents = list()

for AB in ALL_As_Bs:
	AB_data = getFileData(AB)
	AB_data = AB_data.decode('latin-1')
	AB_data = re.sub(r"[^A-Za-z0-9%]"," ",AB_data)
	raw_documents.append(AB_data)
print "raw docs done"
gen_docs = [[w.lower() for w in word_tokenize(text)] for text in raw_documents]

# cPickle.dump(gen_docs, open('./gen_docs', 'wb')) 
# save_pickle(gen_docs,"gen_docs")
print "gen_docs done"

dictionary = gensim.corpora.Dictionary(gen_docs)

dictionary.save("../pickle_files/dictionary")
print "dictionary dont"

corpus = [dictionary.doc2bow(gen_doc) for gen_doc in gen_docs]

tf_idf = gensim.models.TfidfModel(corpus)

tf_idf.save("../pickle_files/tf_idf")
print "tf-idf done"

sims = gensim.similarities.Similarity('.',tf_idf[corpus],num_features=len(dictionary))

sims.save("../pickle_files/sims")
