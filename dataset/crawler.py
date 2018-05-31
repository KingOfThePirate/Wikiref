from bs4 import BeautifulSoup
import urllib2,wikipedia,sys,re,os,requests,httplib
import Queue

def count_aphabets(str1):
	count = 0
	for char in str1:
		if char.isalpha() :
			count = count + 1
	return count

def trim_ref(str1):
	count = 0
	for char in str1:
		if not (char.isupper() or char == "\"") :
			count = count + 1
		else :
			break
	return count

def wikipedia_contents(soup):
	content_tag = soup.find("div", {"id": "content"})

	if content_tag is None:
		return None
	return content_tag.text.encode("utf-8")

def getWikiCategories(soup):
	categories = list()

	categories_div = soup.find("div",{"id":"mw-normal-catlinks"})
	if categories_div is not None:
		categories_link = categories_div.findAll("a")
		for link in categories_link:
			categories.append(link["href"].encode("utf-8").split(":")[-1])
		categories = categories[1:]

	return categories

def getWikiReference(soup,f):

	references = soup.find("span",{"id":"References"})
	if references is None:
		f.write("##References\n")
		return
	references = references.parent
	references = references.find_next_siblings()
	if len(references) == 0:
		f.write("##References\n")
		return
	references = references[0]

	references_children = references.findAll("li")

	f.write("##References\n")
	cite_number = 1
	for reference in references_children :
		reference = reference.text.encode("utf-8")
		index = trim_ref(reference)
		reference = reference[index:]
		
		temp_str1 = re.search( r'[0-9]{4}', reference, re.M|re.I)
		if temp_str1 is None :
			f.write("Year : None$$"+str(cite_number)+"\n")
			index3 = 500
		else:
			temp_str1 = str(temp_str1.group())
			temp_year = str(re.search( r'[0-9]{4}', temp_str1, re.M|re.I).group())
			f.write("Year : "+temp_year+"$$"+str(cite_number)+"\n")
			index3 = reference.find(temp_year)

		if reference.find("\"") > -1:
			index1 = reference.index("\"")
		else :
			index1 = 500
		if reference.find("(") > -1:
			index2 = reference.index("(")
		else :
			index2 = 500

		delimiter = min(index2,index1,index3)

		if delimiter == 500:
			f.write("Authors : None\n")
			f.write("Reference : "+str(reference)+"\n")
		else :
			f.write("Authors : "+reference[0:delimiter]+"\n")
			f.write("Reference : "+reference[delimiter:]+"\n")
		cite_number = cite_number + 1


def printContent(page_title,soup,f):
	
	wiki_content_py = wikipedia_contents(soup)
	div_id_content = soup.find("div",{"class":"mw-parser-output"})

	if wikipedia_contents is None:
		return

	try:
		div_id_content.find("div",{"id":"toc"}).decompose()
	except AttributeError:
		pass
	try:
		div_id_content.find("table",{"class":"vertical-navbox nowraplinks plainlist"}).decompose()
	except AttributeError:
		pass
	try:
		div_id_content.find("table",{"class":"plainlinks metadata ambox ambox-content ambox-multiple_issues compact-ambox"}).decompose()
	except AttributeError:
		pass
	try:
		div_id_content.find("table",{"role":"presentation"}).decompose()
	except AttributeError:
		pass

	if div_id_content is not None:
		a_tags = div_id_content.findAll('a',href=True);
		wiki_links_page_title = list()
		for a_tag in a_tags:
			if a_tag['href'][0:6] == "/wiki/" and len(a_tag.text.encode("utf-8")) > 2:
				wiki_links_page_title.append(a_tag)
		
		start_index = 0
		for link in wiki_links_page_title:
			text = link.text.encode("utf-8")
			link = link['href'][6:].encode("utf-8")
			found_at = wiki_content_py.find(text,start_index)
			if found_at != -1:
				wiki_content_py = 	wiki_content_py[:found_at] + "[[" + link + "||" + wiki_content_py[found_at:found_at+len(text)] + "]]" + wiki_content_py[found_at+len(text):]
			start_index = found_at + 6 + len(link) + len(text)

	lines = wiki_content_py.splitlines()

	for line in lines:
		if count_aphabets(line) > 3 and not(line.find("{") != -1  and line.find("}") != -1):
			f.write(str(line)+"\n")

	return wiki_links_page_title

def getWikiContents(line):

	if line[-5:] == ".wiki":
		line = line[:-5]
	#If file already exists
	if os.path.isfile(line+".wiki"):
		return
	#Create the file
	f = open(line+".wiki", 'w')

	wiki_main = "https://en.wikipedia.org"
	folder = "/wiki/"+line
	data_read = urllib2.urlopen(wiki_main+folder).read()
	soup = BeautifulSoup(data_read,'html.parser')

	f.write("##Title:"+str(line)+"\n")

	#Print Content
	f.write("##Content:\n")

	wiki_cat = printContent(line,soup,f)

	f.write("##Categories:\n")
	categories = getWikiCategories(soup)
	if categories is not None:
		for category in categories :
			f.write(str(category.encode("utf-8"))+"\n")

	f.write("##IntraLinks:\n")
	try:
		if wiki_cat is not None:
			for link in wiki_cat:
				text = link.text.encode("utf-8")
				link = link['href'][6:].encode("utf-8")
				f.write(link+"\n")
	except KeyError as e:
		pass

	f.write("##External References Links:\n")

	getWikiReference(soup,f)

	f.close()


error_file = open("error","a")
list_of_wikis = open(sys.argv[1],"r").read().splitlines()

for line in list_of_wikis:
	if line[0:4] == "[[[[":
		continue
	if line[0:2] == "[[":
		line = line[2:]
	if line is None or len(line) == 0:
		continue

	while True:
		try:
			getWikiContents(line)
		except urllib2.URLError:
			if os.path.isfile(line+".wiki"):
				try:
					os.remove(line+".wiki")
				except Exception:
					pass
			continue
		except requests.exceptions.ProxyError:
			if os.path.isfile(line+".wiki"):
				try:
					os.remove(line+".wiki")
				except Exception:
					pass
			continue
		except httplib.IncompleteRead:
			if os.path.isfile(line+".wiki"):
				try:
					os.remove(line+".wiki")
				except Exception:
					pass
			continue
		except  wikipedia.exceptions.DisambiguationError:
			if os.path.isfile(line+".wiki"):
				try:
					os.remove(line+".wiki")
				except Exception:
					pass
			break
		break
error_file.close()