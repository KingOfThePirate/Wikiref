def getIntraLinks(file_name,path):
	try:
		file_lines = open(path+file_name,"r").read().splitlines()
	except Exception:
		print file_name
		return []
	try:
		intralinks_index = file_lines.index("##IntraLinks:")
		external_links_index = file_lines.index("##External References Links:")
	except ValueError:
		return []
	return file_lines[intralinks_index+1:external_links_index]

wikis = open("3842","r").read().splitlines()

bothab = open("BothAnB","a")

for wiki in wikis:
	bothab.write(wiki[-5:]+"\n")
	intra_list = getIntraLinks(wiki,"../all_12_parts/")
	for intra in intra_list:
		if ":" not in intra:
			bothab.write(intra+"\n")
bothab.close()