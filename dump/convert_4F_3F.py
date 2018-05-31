import os,sys

def getFileData(filename,parent="./"):
	fp = open(parent+filename,"r")
	data = fp.read()
	fp.close()
	return data

file_4k = getFileData("train_data_SVMRank_4F_ordered.dat").splitlines()

file_3k = open("train_data_SVMRank_3F.dat","w")

for line in file_4k:
	tokens = line.split()
	file_3k.write(" ".join(tokens[:-1])+"\n")

file_3k.close()