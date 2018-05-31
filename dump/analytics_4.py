# importing csv module
import csv,os,collections
import matplotlib.pyplot as plt
import numpy as np

# csv file name
filename = "count_3_3.csv"
 
# initializing the titles and rows list
fields = []
rows = []
columns = [[],[],[],[],[],[],[]]
 
# reading csv file
with open(filename, 'r') as csvfile:
    # creating a csv reader object
    csvreader = csv.reader(csvfile, delimiter = ';')
     
    # extracting field names through first row
    fields = csvreader.next()
 
    # extracting each data row one by one
    for row in csvreader:
        if "TLDR.wiki" in row:
            rows.append(['TL;DR.wiki','1','3215','8','1','71','1'])
        else:
            rows.append(row)
 
    #Getting the columns
    for row in rows:
        for j in range(0,7):
            # print row
            columns[j].append(row[j])


rows_z = []
rows_nz = []

for row in rows:
	if row[1] == '0':
		rows_z.append(row)
	else:
		rows_nz.append(row)

columns_nz = [[],[],[],[],[],[],[]]

for row in rows_nz:
    for j in range(0,7):
    	columns_nz[j].append(row[j])

columns_nz[1] = results = map(int, columns_nz[1])
columns_nz[2] = results = map(int, columns_nz[2])
columns_nz[3] = results = map(int, columns_nz[3])
columns_nz[4] = results = map(int, columns_nz[4])
columns_nz[5] = results = map(int, columns_nz[5])
columns_nz[6] = results = map(int, columns_nz[6])

columns_nz_1by3 = [x/(y*z*1.0) for x, y, z in zip(columns_nz[1], columns_nz[3],columns_nz[5])]
columns_nz_4by5 = [x/(y*1.0) for x, y in zip(columns_nz[4], columns_nz[5])]
columns_nz_6by3 = [x/(y*1.0) for x, y in zip(columns_nz[6], columns_nz[3])]

overlap_1_dict = dict()

for i in columns_nz[1]:
	if i not in overlap_1_dict.keys():
		overlap_1_dict[i] = columns_nz[1].count(i)

plt.rcParams.update({'font.size': 22})


filter_over_05 = open("over_05_5","w")
dirty = open("over_05_dirty_5","w")
hop_5_cat = open("5_hop_cat","r").read().splitlines()
hop_5_cat = set(hop_5_cat)
error_f = open("error_f_5","w")

for x in xrange(0,len(columns_nz_6by3)):
	if columns_nz_6by3[x] < 0.5:
		continue
	wiki = columns_nz[0][x]
	wiki_f = open("../all_12/"+wiki,"r").read().splitlines()
	try:
		intralinks_index = wiki_f.index("##Categories:")
		external_links_index = wiki_f.index("##IntraLinks:")
	except ValueError:
		error_f.write(wiki+"\n")
		continue
	cats_wiki = wiki_f[intralinks_index+1:external_links_index]
	if len(cats_wiki) == 0 or cats_wiki is None:
		error_f.write(wiki+"\n")
		continue
	if len(hop_5_cat.intersection(set(cats_wiki))) > 0:
		filter_over_05.write(wiki+"\n")
	else:
		dirty.write(wiki+"\n")


error_f.close()
filter_over_05.close()
