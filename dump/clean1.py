import os

f = open("wiki_and_cats6","r").read().splitlines()
w = open("wiki_and_cats","a")
for line in f:
	if line[0:5] == "*****":
		index = line.rfind('*')
		w.write(line[0:index+1]+'\n'+line[index+1:]+'\n')
	else:
		w.write(line+"\n")

w.close()