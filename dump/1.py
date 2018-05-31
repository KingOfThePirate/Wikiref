from bs4 import BeautifulSoup
import urllib2,wikipedia,sys,re,os,requests,httplib
import Queue 
###################### My functions #############################
def count_aphabets(str1):
	count = 0
	for char in str1:
		if char.isalpha() :
			count = count + 1
	return count
def diffList(first, second):
        second = set(second)
        return [item for item in first if item not in second]

def trim_ref(str1):
	count = 0
	for char in str1:
		if not (char.isupper() or char == "\"") :
			count = count + 1
		else :
			break
	return count

def getWikiCategories(page_title):
	wiki_main = "https://en.wikipedia.org"
	folder = "/wiki/"+page_title
	data_read = urllib2.urlopen(wiki_main+folder).read()
	soup = BeautifulSoup(data_read,'html.parser')

	categories = list()

	categories_div = soup.find("div",{"id":"mw-normal-catlinks"})
	if categories_div is not None:
		categories_link = categories_div.findAll("a")
		for link in categories_link:
			categories.append(link["href"].encode("utf-8").split(":")[-1])
		categories = categories[1:]

	return categories
def printContent(page_title):
	# page_title = "Information_retrieval"
	url_wiki = urllib2.unquote(page_title).decode('utf8')
	wiki_main = "https://en.wikipedia.org"
	folder = "/wiki/"+page_title
	data_read = urllib2.urlopen(wiki_main+folder).read()
	soup = BeautifulSoup(data_read,'html.parser')
	wiki_content_py = wikipedia.WikipediaPage(url_wiki).content.encode("utf-8")
	div_id_content = soup.find("div",{"class":"mw-parser-output"})

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
			start_index = found_at + 4 + len(link)

	lines = wiki_content_py.splitlines()

	for line in lines:
		if count_aphabets(line) > 3 and not(line.find("{") != -1  and line.find("}") != -1):
			print line
def getWikiContents(page_title):
	print page_title
	
	if os.path.isfile(page_title+".wiki"):
		return

	url_wiki = urllib2.unquote(page_title).decode('utf8')
	topic_page = wikipedia.WikipediaPage(url_wiki)

	orig_stdout = sys.stdout
	f = open(page_title+".wiki", 'w')
	sys.stdout = f

	print "##Title:%s" %(topic_page.title.encode("utf-8"))

	print "##Content:"
	printContent(page_title)
	# mycontent = topic_page.content.encode("utf-8")
	# mycontent = mycontent.splitlines()
	# for line in mycontent :
	# 	if count_aphabets(line) > 3 and not(line.find("{") != -1  and line.find("}") != -1):
	# 		print line

	print "##Categories:"
	categories = getWikiCategories(page_title)
	for category in categories :
		print "%s" %(category.encode("utf-8"))

	print "##IntraLinks:"
	try:
		links = topic_page.links
		for link in links :
			print link.encode("utf-8")
	except KeyError as e:
		pass

	print "##External References Links:"
	try:
		references = topic_page.references
		for link in references :
			print link.encode("utf-8")
	except KeyError as e:
		pass

	sys.stdout = orig_stdout
	f.close()
	getWikiReference(page_title)

def getWikiReference(page_title):
	orig_stdout = sys.stdout
	f = open(page_title+".wiki", 'a')
	sys.stdout = f


	wiki_main = "https://en.wikipedia.org"
	sample1 = "/wiki/"+page_title

	data_read = urllib2.urlopen(wiki_main+sample1).read()

	soup = BeautifulSoup(data_read,'html.parser')
	# references = soup.find("ol",{"class":"references"})
	# if references is None:
	# 	references = soup.find("span",{"id":"References"})
	# 	references = references.parent
	# 	references = references.find_next_siblings("ul")
	references = soup.find("span",{"id":"References"})
	if references is None:
		print "##References"
		return
	references = references.parent
	references = references.find_next_siblings()
	if len(references) == 0:
		print "##References"
		return
	references = references[0]

	references_children = references.findAll("li")

	print "##References"
	for reference in references_children :
		reference = reference.text.encode("utf-8")
		index = trim_ref(reference)
		reference = reference[index:]
		
		temp_str1 = re.search( r'[0-9]{4}', reference, re.M|re.I)
		if temp_str1 is None :
			print "Year : None"
			index3 = 500
		else:
			temp_str1 = str(temp_str1.group())
			temp_year = str(re.search( r'[0-9]{4}', temp_str1, re.M|re.I).group())
			print "Year :",temp_year
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
			print "Authors : None"
			print "Reference :",reference
		else :
			print "Authors :",reference[0:delimiter]
			print "Reference :",reference[delimiter:]

	sys.stdout = orig_stdout
	f.close()

def getWikiSubcategoriesAndPages(category):
	wiki_main = "https://en.wikipedia.org"
	folder = "/wiki/Category:"+category
	data_read = urllib2.urlopen(wiki_main+folder).read()

	soup = BeautifulSoup(data_read,'html.parser')

	subcategories = list()
	pagesincategory = list()
	categories = list()

	# subcategories_div = soup.find("div",{"id":"mw-subcategories"})
	# if subcategories_div is not None:
	# 	subcategories_link = subcategories_div.findAll("a")
	# 	for link in subcategories_link:
	# 		subcategories.append(link["href"].encode("utf-8").split(":")[1])

	pagesincategory_div = soup.find("div",{"id":"mw-pages"})
	if pagesincategory_div is not None:
		pagesincategory_div = pagesincategory_div.find("div",{"class":"mw-content-ltr"})
		pagesincategory_link = pagesincategory_div.findAll("a")
		for link in pagesincategory_link:
			pagesincategory.append(link["href"].encode("utf-8").split("/")[-1])

	# categories_div = soup.find("div",{"id":"mw-normal-catlinks"})
	# if categories_div is not None:
	# 	categories_link = categories_div.findAll("a")
	# 	for link in categories_link:
	# 		categories.append(link["href"].encode("utf-8").split(":")[-1])
	# 	categories = categories[1:]

	return pagesincategory
##################### End ########################################

#list_39000 = open("1","r")
#list_39000 = list_39000.read().splitlines()
#list_39000 = list_39000[::-1]
for line in list_39000:
	# line = line[0:-1]
	# getWikiContents(line)
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
				os.remove(line+".wiki")
			continue
		except requests.exceptions.ProxyError:
			if os.path.isfile(line+".wiki"):
				os.remove(line+".wiki")
			continue
		except httplib.IncompleteRead:
			if os.path.isfile(line+".wiki"):
				os.remove(line+".wiki")
			continue
		except  wikipedia.exceptions.DisambiguationError:
			if os.path.isfile(line+".wiki"):
				os.remove(line+".wiki")
			break
		except:
			if os.path.isfile(line+".wiki"):
				os.remove(line+".wiki")
			break
		break
	print line
