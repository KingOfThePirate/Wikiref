import os

rows = open("features.csv","r").read().splitlines()[1:]

yes = open("features_yes.csv","w")
no = open("features_no.csv","w")

for row in rows:
	if row.split("#")[-1] == "N":
		no.write(row+"\n")
	else:
		yes.write(row+"\n")

yes.close()
no.close()