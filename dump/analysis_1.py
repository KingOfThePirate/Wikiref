import os,sys

filename = sys.argv[1]

As = open(filename,"r").read().split("@@NewA@@")[1:]

yes_ref = list()
no_ref = list()

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
			# print A_name,B_name 
			yes_ref.append(0.8)
		else:
			yes_ref.append(round((sum(b_ref_y))/len(b_ref_y),2))
		if len(b_ref_n)!=0 :
			no_ref.append(round((sum(b_ref_n))/len(b_ref_n),2))
		else:
			no_ref.append(0)

yes_ref.sort()
no_ref.sort()
if len(yes_ref) != len(no_ref):
	print "Something is wrong"

print "max",yes_ref[-1]
print "min",yes_ref[0]
print "average",sum(yes_ref)/len(yes_ref)
print "median",yes_ref[(len(yes_ref)/2)+1]
print ""
print "max",no_ref[-1]
print "min",no_ref[0]
print "average",sum(no_ref)/len(no_ref)
print "median",no_ref[(len(no_ref)/2)+1]