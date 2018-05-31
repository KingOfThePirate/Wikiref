import os

f = "./count_3_1.csv"

f = open(f,"r").read().splitlines()

zero_f = open("zero_overlap","w")
nonzero_f = open("nonzero_overlap","w")
for f_i in f:
	f_i = f_i.split(";")
	# print f_i[1]
	if f_i[1] == "0":
		zero_f.write(f_i[0]+"\n")
	else:
		nonzero_f.write(f_i[0]+"\n")