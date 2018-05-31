import os,sys

def parseAB(text):
	if len(text)==0 or text is None:
		return None
	lines = text.splitlines()
	A = lines[0][2:].split("##__##")[0]
	return A

print "Hello"
AB_overlap_relation = open("AB_overlap_relation","r").read().split("@@New@@\n")[1:]
As_geq05 = open("wiki_pages_in_phy_5_As_geq05","r").read().splitlines()
AB_overlap_relation_geq05 = open("AB_overlap_relation_geq05","w")

for each in AB_overlap_relation:
	parse_a = parseAB(each)
	if parse_a in As_geq05:
		AB_overlap_relation_geq05.write("@@New@@\n"+each)
AB_overlap_relation_geq05.close()