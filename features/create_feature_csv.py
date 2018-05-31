import os,sys

features_files = ["Step1/tfidf_summary_feature","Step1/outsent_feature","Step1/insent_feature","Step1/out_degree_feature","Step1/in_degree_feature","Step1/summaryVec_feature","Step1/outsentVec_feature","Step1/insentVec_feature"]
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
				features_dict[wiki][token[1]][token[0]].append(round(float(token[2]),3))

for x in range(1,features_len):
	lines_in_file = open(features_files[x],"r").read().splitlines()
	wiki = ""
	for line in lines_in_file:
		if (line[:2] == "@@" and line[-2:] == "@@"):
			wiki = line[2:-2]
		else:
			token = line.split("##")
			if len(token) == 3:
				if len(features_dict[wiki][token[1]][token[0]]) == x:
					features_dict[wiki][token[1]][token[0]].append(round(float(token[2]),3))

feature_writer = open("features.csv","w")

for key1, value1 in features_dict.iteritems():
	feature_writer.write(key1+"\n")
	for key2,value2 in features_dict[key1].iteritems():
		for key3,value3 in features_dict[key1][key2].iteritems():
			if len(features_dict[key1][key2][key3]) == features_len:
				feature_writer.write(str(key3)+"#")
				for f_i in features_dict[key1][key2][key3]:
					feature_writer.write(str(f_i)+"#")
				feature_writer.write(str(key2)+"\n")

feature_writer.close()
