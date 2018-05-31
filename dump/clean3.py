import os

features_files = ["in_sent_analysis1","out_sent_analysis1"]
features_len = len(features_files)
features_dict = dict()

for files in features_files[:1]:
	lines_in_file = open(files,"r").read().splitlines()
	wiki = ""
	for line in lines_in_file:
		if (line[:2] == "@@" and line[-2:] == "@@"):
			wiki = line[2:-2]
			features_dict[wiki] = {'Y':{},'N':{}}
		else:
			token = line.split("##")
			if len(token) == 3:
				features_dict[wiki][token[1]][token[0]] = list()
				features_dict[wiki][token[1]][token[0]].append(token[2])
keys1 = list(features_dict.keys())
print len(keys1)
features_dict = dict()

for files in features_files[1:]:
	lines_in_file = open(files,"r").read().splitlines()
	wiki = ""
	for line in lines_in_file:
		if (line[:2] == "@@" and line[-2:] == "@@"):
			wiki = line[2:-2]
			features_dict[wiki] = {'Y':{},'N':{}}
		else:
			token = line.split("##")
			if len(token) == 3:
				features_dict[wiki][token[1]][token[0]] = list()
				features_dict[wiki][token[1]][token[0]].append(token[2])

keys2 = list(features_dict.keys())
print len(keys2)

left_key = list(set(keys2) - set(keys1))

filew = open("left_key","w")

for key in left_key:
	filew.write(key+"\n")
filew.close()