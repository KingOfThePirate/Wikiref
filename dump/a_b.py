import jellyfish,os
######## Functions ###########
def getRef(name,parent):
	try:
		curr = open(parent+name,"r")
	except IOError:
		not_downloaded.write(name+"\n")
		return None
	curr_lines = curr.read().splitlines()
	curr.close()
	try :
		ref_index = curr_lines.index("##References")
		all_ref = curr_lines[ref_index+1:]
		list_of_ref_curr = list()
		len_all_ref = len(all_ref)
		if len_all_ref%3 is not 0:
			not_downloaded.write(name+"\n")
			return None
		for j in xrange(0,len_all_ref,3):
			list_of_ref_curr.append({'year':all_ref[j].split(':', 1 )[1],'authors':all_ref[j+1].split(':', 1 )[1],'ref':all_ref[j+2].split(':', 1 )[1]})
		return list_of_ref_curr
	except ValueError:
		not_downloaded.write(name+"\n")
		return None
	except IndexError:
		not_downloaded.write(name+"\n")
		return None
######## 	End	   ###########

a = getRef("Recurrent_neural_network.wiki","../wiki_links/all_12_parts/")
b = getRef("Long_short-term_memory.wiki","../wiki_links/all_12_parts/")

for a_i in a:
	for b_i in b:
		a_au = unicode(a_i['authors'],"utf-8")
		b_au = unicode(b_i['authors'],"utf-8")
		a_ref = unicode(a_i['ref'],"utf-8")
		b_ref = unicode(b_i['ref'],"utf-8")
		if jellyfish.jaro_winkler(a_ref, b_ref) > 0.8:
			print "###"
			print a_i
			print b_i