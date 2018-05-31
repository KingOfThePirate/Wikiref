# importing csv module
import csv,os,collections
import matplotlib.pyplot as plt
import numpy as np

# csv file name
filename = "count_3_1.csv"
 
# initializing the titles and rows list
fields = []
rows = []
columns = [[],[],[],[],[],[]]
 
# reading csv file
with open(filename, 'r') as csvfile:
    # creating a csv reader object
    csvreader = csv.reader(csvfile, delimiter = ';')
     
    # extracting field names through first row
    fields = csvreader.next()
 
    # extracting each data row one by one
    for row in csvreader:
        rows.append(row)
 
    #Getting the columns
    for row in rows:
    	for j in range(0,6):
    		columns[j].append(row[j])

rows_z = []
rows_nz = []

for row in rows:
	if row[1] == '0':
		rows_z.append(row)
	else:
		rows_nz.append(row)

columns_nz = [[],[],[],[],[],[]]

for row in rows_nz:
    for j in range(0,6):
    	columns_nz[j].append(row[j])

columns_nz[1] = results = map(int, columns_nz[1])
columns_nz[2] = results = map(int, columns_nz[2])
columns_nz[3] = results = map(int, columns_nz[3])
columns_nz[4] = results = map(int, columns_nz[4])
columns_nz[5] = results = map(int, columns_nz[5])

columns_nz_1by3 = [x/(y*z*1.0) for x, y, z in zip(columns_nz[1], columns_nz[3],columns_nz[5])]
columns_nz_4by5 = [x/(y*1.0) for x, y in zip(columns_nz[4], columns_nz[5])]

overlap_1_dict = dict()

for i in columns_nz[1]:
	if i not in overlap_1_dict.keys():
		overlap_1_dict[i] = columns_nz[1].count(i)

plt.rcParams.update({'font.size': 22})

# plt.plot(overlap_1_dict.keys(), overlap_1_dict.values(),"ro")
# plt.xlabel('Overlaps')
# plt.ylabel('Number Documents')
# plt.title('My first graph!')
# plt.grid(True)
# plt.ylim((0,100))
# plt.show()

# binwidth = 5
# plt.hist(columns_nz[1], normed=False, bins=range(min(columns_nz[1]), max(columns_nz[1]) + binwidth, binwidth))
# plt.ylabel('Number of Documents')
# plt.show()

# binwidth = 10
# plt.hist(columns_nz[3], normed=False, bins=range(min(columns_nz[3]), max(columns_nz[3]) + binwidth, binwidth))
# plt.ylabel('Number of Documents')
# plt.ylim()
# plt.show()

# binwidth = 0.01
# plt.hist(columns_nz_1by3, normed=False, bins=np.arange(min(columns_nz_1by3), max(columns_nz_1by3) + binwidth, binwidth))
# plt.ylabel('Number of Documents')
# plt.show()

# plt.scatter(columns_nz[1],columns_nz[5])
# plt.xlabel('Overlap')
# plt.ylabel('Number of Wiki_links')
# plt.grid(True)
# plt.show()

binwidth = 0.1
plt.hist(columns_nz_4by5, normed=False, bins=np.arange(min(columns_nz_4by5), max(columns_nz_4by5) + binwidth, binwidth))
plt.ylabel('Number of A\'s')
plt.xlabel('Fraction of B contributing')
plt.show()