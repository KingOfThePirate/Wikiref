from bs4 import BeautifulSoup
import urllib2,wikipedia,sys,re,os,requests,httplib

parent = "./"
files = os.listdir(parent)

for file in files:
	if file[-3:] != "txt":
		continue
	file_open = open(parent+file,"r")
	file_w = open(parent+file+"_new","w")
	for a in file_open:
		a = a[:-1]
		last_hash = a.rfind("#")
		if last_hash is not -1:
			a = a[0:last_hash]
		if a.find("[") == -1 and a.find("Special:") == -1:
			file_w.write(a+"\n")