import jellyfish,os,sys,re


def getRef(name,parent=dataset_path):
	try:
		curr = open(parent+name,"r")
	except IOError:
		print "IOError getRef",name
		return None
	curr_lines = curr.read().splitlines()
	curr.close()
	try :
		ref_index = curr_lines.index("##References") + 1
		len_curr_lines = len(curr_lines)
		dict_of_ref_curr = dict()
		count = 0;
		for j in xrange(ref_index,len_curr_lines,1):
			curr_line_split = curr_lines[j].split(":",1)
			if len(curr_line_split) <= 1:
				continue
			curr_line_split[0] = curr_line_split[0].strip(' ')
			curr_line_split[1] = curr_line_split[1].strip(' ')
			if "Year" == curr_line_split[0]:
				year = curr_line_split[1].split("$$")[0]
				number = int(curr_line_split[1].split("$$")[1])
				count = count + 1
			elif "Authors" == curr_line_split[0]:
				author = curr_line_split[1]
				count = count + 1
			elif "Reference" == curr_line_split[0]:
				ref = curr_line_split[1]
				if count == 2 :
					dict_of_ref_curr[number] = {'year':year,'authors':author,'ref':ref}
					count = 0
		return dict_of_ref_curr
	except Exception:
		print "getRef",name
		print sys.exc_info()[0]
		return None
def getWikiFileForamt(name):
	name = name.replace(" ","_")
	name = name.replace("/","$$$")
	if len(name) < 5:
		return name + ".wiki"
	name = name.split("#")[0]
	if name[-5:] != ".wiki":
		name = name + ".wiki"
	return name

def getIntraLinks(file_name,path=dataset_path):
	file_name = file_name.replace(" ","_")
	if file_name[-5:] != ".wiki":
		file_name = file_name + ".wiki"
	try:
		file_lines_fp = open(path+file_name,"r")
	except IOError:
		print file_name
		return []
	file_data = file_lines_fp.read()	
	wikilinks = re.findall(r'\[\[[^ ]{1,}\|\|',file_data)	
	wikilinks = [ getWikiFileForamt(i[2:-2].split('#',1)[0]) for i in wikilinks ] 	
	return wikilinks

dataset_path = "../dataset/ComputerScience/"
list_of_As = "../dataset/ComputerScience_As"

As_fp = open(list_of_As,"r")
As = As_fp.read().splitlines()
As_fp.close()
As.sort()


AB_statistics = open("AB_statistics","w")
AB_overlap_relation = open("AB_overlap_relation","w")

Asgeq05 = open("wiki_pages","w")

dict_AB_ref_overlap = dict()

As_counter = 0
for A in As:
	A = getWikiFileForamt(A)
	if ":" in A:
		continue
	ref_A = getRef(A)
	if ref_A is None or len(ref_A) == 0:
		continue

	dict_AB_ref_overlap[A] = dict()
	Bs = getIntraLinks(A)

	total_ref_A = len(ref_A)
	total_ref_B = 0
	total_ref_overlap_A = 0
	no_Bs = len(Bs)
	no_Bs_overlap = 0
	ref_overlaps = set()

	AB_overlap_relation.write("@@New@@\n")

	for B in Bs:
		if ":" in B:
			continue
		B = getWikiFileForamt(B)
		ref_B = getRef(B)
		if ref_B is None or len(ref_B) == 0:
			continue
		total_ref_B = total_ref_B + len(ref_B)
		dict_AB_ref_overlap[A][B] = list()
		B_overlap_YN = False
		for a_key in ref_A:
			for b_key in ref_B:
				a_au = unicode(ref_A[a_key]['authors'],"utf-8")
				b_au = unicode(ref_B[b_key]['authors'],"utf-8")
				a_ref = unicode(ref_A[a_key]['ref'],"utf-8")
				b_ref = unicode(ref_B[b_key]['ref'],"utf-8")
				jaro_au = jellyfish.jaro_winkler(a_au, b_au)
				jaro_ref = jellyfish.jaro_winkler(a_ref, b_ref)
				if (jaro_au >= 0.75 and len(a_au)>5 and len(b_au)>5) and (jaro_ref >= 0.75 and len(a_ref)>5 and len(b_ref)>5):
					ref_overlaps.add(a_key)
					dict_AB_ref_overlap[A][B].append((a_key,b_key))
					B_overlap_YN = True
					break
		if B_overlap_YN:
			AB_overlap_relation.write("##"+A+"##__##"+B+":YES\n")
			no_Bs_overlap = no_Bs_overlap + 1
		else:
			AB_overlap_relation.write("##"+A+"##__##"+B+":NO\n")

	AB_statistics.write(",".join([str(A),str(total_ref_A),str(total_ref_B),str(len(ref_overlaps)),str(no_Bs),str(no_Bs_overlap)]) +   "\n")
	if (len(ref_overlaps)*1.0)/(total_ref_A*1.0) >= 0.5:
		Asgeq05.write(A+"\n")

	As_counter = As_counter + 1

	if As_counter%100 == 0:
		AB_overlap_relation.close()
		AB_statistics.close()
		Asgeq05.close()
		AB_statistics = open("AB_statistics","a")
		AB_overlap_relation = open("AB_overlap_relation","a")	
		Asgeq05 = open("wiki_pages","a")
	print A

AB_overlap_relation.close()
AB_statistics.close()
Asgeq05.close()
