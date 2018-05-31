import os

def getList(line):
	len_str = len(line)
	a = list()
	b = list()
	for i in xrange(0,len_str):
		if "[" == line[i]:
			if len(a) == len(b):
				a.append(i)
			else:
				a = a[:-1]
				a.append(i)
		if "]" == line[i]:
			b.append(i)
	if len(a) == len(b):
		return [a,b]
	else :
		return [list(),list()]

files_curr = os.listdir(".")
files_curr.sort()

for file in files_curr:
	if file[-4:] == "_cps":
		file_lines = open(file,"r").read().splitlines()
		file_write = open(file[:-4]+"_ucps","w")
		for file_line in file_lines:
			print_line = file_line
			opening_b,closing_b = getList(file_line)
			print getList(file_line)
			diff = 0
			for b_i in xrange(0,len(opening_b)):
				print print_line
				print len(print_line)
				print opening_b[b_i]-diff
				print closing_b[b_i]+1-diff
				print_line = print_line[0:opening_b[b_i]-diff] + print_line[closing_b[b_i]+1-diff:]
				diff += closing_b[b_i] - opening_b[b_i] + 1
				print print_line
				print len(print_line)
			file_write.write(print_line+"\n")
		file_write.close()

		