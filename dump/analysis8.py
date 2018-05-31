import os,editdistance

# Function Start
def getRef(name):
	parent = "./all_12/"
	curr = open(parent+name,"r")
	curr_lines = curr.read().splitlines()
	curr.close()
	try :
		ref_index = curr_lines.index("##References")
		all_ref = curr_lines[ref_index+1:]
		list_of_ref_curr = list()
		len_all_ref = len(all_ref)
		if len_all_ref%3 is not 0:
			return None
		for j in xrange(0,len_all_ref,3):
			list_of_ref_curr.append({'year':all_ref[j].split(':', 1 )[1],'authors':all_ref[j+1].split(':', 1 )[1],'ref':all_ref[j+2].split(':', 1 )[1]})
		return list_of_ref_curr
	except ValueError:
		return None
	except IndexError:
		return None
# Function Stop

parent = "./all_12/"
wiki_citis = "./wiki_citis/"
wiki_less6 = "./wiki_less6/"
wiki_more6 = "./wiki_more6/"

citis = open(wiki_citis+"citis8.txt","w")
less6 = open(wiki_less6+"less68.txt","w")
more6 = open(wiki_more6+"more68.txt","w")

list_files = os.listdir(parent)

list_files.sort()

division = len(list_files)/8
for i in xrange(division*7,len(list_files)):
	curr_ref = getRef(list_files[i])
	citis.write("********"+list_files[i]+"********\n")
	if curr_ref is None:
		continue
	for each_wiki in list_files:
		if each_wiki is list_files[i]:
			continue
		ref_each_wiki = getRef(each_wiki)
		if ref_each_wiki is None:
			continue
		print_citis = list()
		for each_curr_ref in curr_ref:
			for each_ref_each_wiki in ref_each_wiki:
				if each_curr_ref['year'] == each_ref_each_wiki['year']:
					edit_dis_authors = editdistance.eval(each_curr_ref['authors'],each_ref_each_wiki['authors'])
					edit_dis_ref = editdistance.eval(each_curr_ref['ref'],each_ref_each_wiki['ref'])
				else :
					continue
				if edit_dis_authors < 6 or edit_dis_ref < 6:
					print_citis.append(each_wiki)
					# less6.write(str(each_curr_ref)+"\n"+str(each_ref_each_wiki)+"\n"+"{"+str(int(edit_dis_authors))+","+str(int(edit_dis_ref))+"}\n")
	citis.write(str(set(print_citis))+"\n")