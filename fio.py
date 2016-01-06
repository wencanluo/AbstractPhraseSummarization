## @package fio# Package for I/O# @file fio.py# @author Wencan Luo (wencan@cs.pitt.edu)# @date 2011-09-25import types as Typesimport sysimport jsonimport osimport shutilimport codecsdef NewPath(path):	if not os.path.exists(path):		os.makedirs(path)def remove(file):	if IsExist(file):		os.remove(file)def IsExist(file):	return os.path.isfile(file)def IsExistPath(path):	return os.path.exists(path)def DeleteFolder(path):	try:		shutil.rmtree(path)	except Exception:		pass# 	for dirpath, dirnames, filenames in os.walk(path):# 		try:# 			os.rmdir(dirpath)# 		except OSError as ex:# 			print(ex)def ReadFile(file):	"""Input a file, and return a list of sentences		@param file: string, the input file path	@return: list of lines. Note: each line ends with "\r\n" or "\n"	"""		#read the file	f = open(file,'r')	lines = f.readlines()	f.close()		return lines	#sentences = []	#for line in lines:		#get ride of the '\r\n' in the end	#	line = line.rstrip('\r\n')	#	sentences.append(line)		#return sentences		def SaveList(List, file, linetag="\n"):	"""	Save a list into a file. Each item is a line	"""	reload(sys)	sys.setdefaultencoding('utf8')		f = open(file, "w")	for item in List:		f.write(str(item))		f.write(linetag)	f.flush()	f.close()def LoadList(file):	return [line.strip() for line in ReadFile(file)]	def SaveText(text, file):	"""	Save a string into a file. Each item is a line	"""	f = open(file, "w")	f.write(str(text))	f.flush()	f.close()def PrintListwithName(list, name = None):	print name, "\t",	for entry in list:		print entry, "\t",	print	def PrintList(list, sep = "\t", endflag=True, prefix=''):	"""	@function: print out a list into a file.	@param list: list	@param sep: string, the separator between each item	@param endflag: bool, whether each item is a line	@param prefix: string, the prefix of each iterm   	"""	for i in range(len(list)):		entry = list[i]		entry = prefix+str(entry)+prefix		if i==len(list)-1:			if endflag:				print entry			else:				print entry,		else:			print entry + sep,			def PrintDict(dict, SortbyValueflag = True):	"""	@function: print out a dict in a reverse order of the values, the value of the dict should be numeric 	@param dict: dictionary	"""	reload(sys)	sys.setdefaultencoding('utf8')		if SortbyValueflag:		for key in sorted(dict, key = dict.get, reverse = True):			print str(key) + "\t" + str(dict[key])	else:		for key in sorted(dict):			print str(key) + "\t" + str(dict[key])			def SaveDict(dict, file, SortbyValueflag = False):	"""	@function:save a dict	@param dict: dictionary	"""	SavedStdOut = sys.stdout	sys.stdout = open(file, 'w')		if SortbyValueflag:		for key in sorted(dict, key = dict.get, reverse = True):			print key + "\t" + str(dict[key])	else:		for key in sorted(dict.keys()):			print str(key) + "\t" + str(dict[key])	sys.stdout = SavedStdOut	def SaveDict2Json(dict, file):	with open(file, "w") as fout:		json.dump(dict, fout, indent=2, encoding="utf-8")def LoadDictJson(file):	with open(file, "r") as fin:		dict = json.load(fin)	return dict	def LoadDict(file, type='str'):	"""	@function:load a dict	@return dict: dictionary	"""	body = ReadMatrix(file, False)		if body == None: return None	dict = {}	for row in body:		assert(len(row) == 2)				if type == 'str' or type==str:			dict[row[0]] = row[1]		if type == 'float' or type == float:			dict[row[0]] = float(row[1])		if type == 'int' or type == int:			dict[row[0]] = int(row[1])				return dict	def LoadExcel(file, hasHead = True):	rows = ReadFile(file)	y = len(rows)		if y == 0:		return None		x = len(rows[0].split("\t"))		body = [[None]*x]*y		for i in range(y):		row = rows[i]		cols = row.split("\t")		if len(cols) != x:			print "Excel format is wrong"			print i, x, len(cols)			return None		for j in range(x):			col = cols[j]			body[i][j] = col		if hasHead:		head = body[0]		body = body[1:]		return head, body	else:		return bodydef CRFWriter(file, data):	SavedStdOut = sys.stdout	sys.stdout = open(file, 'w')		for row in data:		for i, col in enumerate(row):			if i == len(row) - 1:				print col,			else:				print str(col)+"\t",		print		sys.stdout = SavedStdOut	def CRFReader(file):	body = ReadMatrix(file, False)		header = ['True','Predict']	body = [row[-2:] for row in body]		return header, body	#types = 'String', 'Category', 'Continuous'def ArffWriter(file, head, types, name, data):	"""	Function: write the data to a arff file for Weka	@param file: string, the output file name	@param head: list, the attribute name list, the class label is "@class@"	@param types: the types of the attributes, an attribute can be 'String', 'Category' or 'Continuous'	@param name: string, the name of the relationship	@param data: matrix, the data, each row is an instance	"""	SavedStdOut = sys.stdout	reload(sys)	sys.setdefaultencoding('utf8')		sys.stdout = open(file, 'w')	print "@relation " + name	print	#get the class categories	n = len(head)	cats = []	for i in range(n):		cats.append({})		for row in data:		for i in range(n):			if types[i] != 'Category': continue			if row[i] == None:				row[i] = ""			if not cats[i].has_key(row[i]):				cats[i][row[i]] = 1				for i in range(n):		att = head[i]		if types[i] == 'Category':			print "@attribute " + att + " {",			#print the class			cat = sorted(cats[i].keys())			PrintList(cat, ",", False, "'")			print "}"		elif types[i] == 'String':			print "@attribute " + att + " string"		elif types[i] == 'Continuous':			print "@attribute " + att + " NUMERIC"		else:			print "Not Supported"	print		print "@data"		#write data	for row in data:		for i in range(len(row)):			atr = row[i]						if atr != None:				if types[i] == 'Continuous':					if i==len(row) - 1:						print str(atr),					else:						print str(atr)+",",				else:					if type(atr) is Types.StringType:						atr = atr.replace("'","\\'")					if type(atr) is Types.UnicodeType:						atr = atr.replace("'","\\'")					if i==len(row) - 1:						print '\''+str(atr)+'\'',					else:						print '\''+str(atr)+'\''+",",			else:				print "'',",		print		sys.stdout = SavedStdOutdef MulanWriter(file, labels, head, types, name, data):	"""	Function: write the data to a arff file for Mulan [http://mulan.sourceforge.net/index.html]	@param filename: string, the output file name without extension	@param labels: list, the labels 	@param head: list, the attribute name list, the class label is "@class@"	@param types: the types of the attributes, an attribute can be 'String', 'Category' or 'Continuous'	@param name: string, the name of the relationship	@param data: matrix, the data, each row is an instance	"""	ArffWriter(file + '.arff', head, types, name, data)		#write the XML	SavedStdOut = sys.stdout	sys.stdout = open(file + '.xml', 'w')		print '<?xml version="1.0" encoding="utf-8"?>'	print '<labels xmlns="http://mulan.sourceforge.net/labels">'		for label in labels:		print '\t<label name="'+label+'"></label>'	print '</labels>'		sys.stdout = SavedStdOutdef MulanOutReader(file, labelnames = None):	lines = ReadFile(file)		body = []		for line in lines:		begin = line.find('[')		end = line.find(']')				if begin == -1 or end == -1:			print "Error"		labels = line[begin+1:end].split(',')				row = [1 if x.strip()=='true' else 0 for x in labels]				body.append(row)			return body	 	def ReadMatrix(file, hasHead=True):	"""	Function: Load a matrix from a file. The matrix is M*N	@param file: string, filename	@param hasHead: bool, whether the file has a header	@return header, body	"""	lines = ReadFile(file)	#print len(lines)		tm = []	for line in lines:		row = []		line = line.strip()		if len(line) == 0: continue		for num in line.split("\t"):			row.append(num.strip())		tm.append(row)	if hasHead:		header = tm[0]		body = tm[1:]		return header, body	else:		return tmdef WriteMatrix(file, data, header=None, tab='\t'):	"""	Function: save a matrix to a file. The matrix is M*N	@param file: string, filename	@param data: M*N matrix,  	@param header: list, the header of the matrix	"""	reload(sys)	sys.setdefaultencoding('utf8')    	SavedStdOut = sys.stdout	sys.stdout = open(file, 'w')		if header != None:		for j in range(len(header)):			label = header[j]			if j == len(header)-1:				print label			else:				print label, tab,	for row in data:		for j in range(len(row)):			col = row[j]			if j == len(row) - 1:				print col			else:				print col, tab,		sys.stdout = SavedStdOutdef WriteCSV(file, data, header=None, tab='\t'):	"""	Function: save a matrix to a file. The matrix is M*N	@param file: string, filename	@param data: M*N matrix,  	@param header: list, the header of the matrix	"""	reload(sys)	sys.setdefaultencoding('utf8')    	SavedStdOut = sys.stdout	sys.stdout = codecs.open(file, 'w', 'utf-8')		if header != None:		for j in range(len(header)):			label = header[j]			if j == len(header)-1:				sys.stdout.write(label+'\n')			else:				sys.stdout.write(label + tab)	for row in data:		for j in range(len(row)):			col = row[j]			if j == len(row) - 1:				sys.stdout.write(col+'\n')			else:				sys.stdout.write(col + tab)		sys.stdout = SavedStdOut	def ExtractWekaScore(input, output):	lines = ReadFile(input)		S = []		header = ['TP Rate', 'FP Rate', 'Precision', 'Recall', 'F-Measure', 'ROC Area']	key = 'Weighted Avg.'	for line in lines:		line = line.strip()		if line.startswith(key):			line = line[len(key):]			scores = line.strip().split()			S.append(scores)		WriteMatrix(output, S, header)if __name__ == '__main__':	labels = ['area', 'food', 'name', 'pricerange', 'addr', 'phone', 'postcode', 'signature']				MulanOutReader('res/dstc2_train_request_actngram_ngram.arff.label', labels)		