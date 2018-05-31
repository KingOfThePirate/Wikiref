import os,sys,re,pickle

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
	new_line = re.sub(r"[^A-Za-z0-9]",	" ",new_line)
	return new_line

file_prefix = sys.argv[1]
no_files = int(sys.argv[2])

file_names = [ file_prefix+str(i) for i  in range(no_files) ]

final_dict = dict()

for i in range(no_files):
	loop_dict = load_pickle(file_names[i])
	for key1 in loop_dict:
		final_dict[key1] = dict()
		for key2 in loop_dict[key1]:
			final_dict[key1][key2] = dict()
			for key3 in loop_dict[key1][key2]:
				final_dict[key1][key2][key3] = clean(loop_dict[key1][key2][key3])


save_pickle(final_dict,file_prefix[:-1])