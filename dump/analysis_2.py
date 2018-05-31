import os,sys

def merge(a,b):
	a = a[::-1]
	b = b[::-1]
	i = 0
	j = 0
	c = list()
	while i < len(a) and j < len(b):
		if a[i] < b[j]:
			c.append("NO")
			j = j + 1
		else:
			c.append("YES")
			i = i + 1
	while i < len(a):
		c.append("YES")
		i = i + 1
	while j < len(b):
		c.append("NO")
		j = j + 1
	return c
def getPrecisionAtK(a,k = 1):
	count  = 0
	k = min(len(a),k)
	for i in range(k):
		if a[i] == "YES":
			count += 1
	return round((count)/(k*1.0),2)

def getRecallAtK(a,number_of_yes,k=1):
	count  = 0
	k = min(len(a),k)
	for i in range(k):
		if a[i] == "YES":
			count += 1
	return round((count)/(number_of_yes*1.0),2)

filename = sys.argv[1]
k = int(sys.argv[2])
As = open(filename,"r").read().split("@@NewA@@")[1:]

prec_at_k = list()
recall_at_k = list()

for A_ in As:
	A = A_.strip('\n')
	A = A.split("@@NewB@@")
	A_name = A[0].split(":",1)[1][:-1]
	for B_ in A[1:]:
		B = B_.strip('\n')
		B = B.splitlines()
		B_name = B[0].split(":",1)[1]
		b_ref_y = list()
		b_ref_n = list()
		for ref_ in B[1:]:
			ref = ref_.split("$$")
			try:
				if ref[-1] == "YES":
					b_ref_y.append(float(ref[1]))
				else:
					b_ref_n.append(float(ref[1]))
			except ValueError:
				print A_name,B_name,ref[1]
				sys.exit()

		if len(b_ref_y)==0:
			continue
		c = merge(b_ref_y,b_ref_n)
		pak = getPrecisionAtK(c,k)
		rak = getRecallAtK(c,len(b_ref_y),k)
		prec_at_k.append(pak)
		recall_at_k.append(rak)

print "Average :",sum(prec_at_k)/len(prec_at_k)
print "Average :",sum(recall_at_k)/len(recall_at_k)