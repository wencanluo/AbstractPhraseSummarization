#!/usr/bin/env python
# encoding: utf-8
#
#  OrigReader.py
#  
#
#  Created by haipiaoxiao on 10-3-20.
#  Copyright (c) 2010 __MyCompanyName__. All rights reserved.
#
# modifed based on arffRawData.py
# Requirement of the input raw data: 
# The first line is the feature name, and the rest of the file will be lines of instances. all items are seperated by certain delimiter (default is "<ITEMBREAK>")
# default settings for preprocessing: 
#	lower cases
#	stemming
#	remove quotation marks
	

from __future__ import division
import nltk, re, pprint, os, operator
from nltk import *
from nltk import wordpunct_tokenize
from nltk.probability import FreqDist
from nltk.text import Text, TextCollection
from nltk.stem.porter import PorterStemmer
from nltk.util import bigrams
from nltk.corpus.reader.plaintext import PlaintextCorpusReader as corpusReader
import xlrd
import numpy as np
#import matplotlib.pyplot as pylab  #doesn't work when upgrading to os 10.7
#from matplotlib.ticker import EngFormatter

def readInData(dir_data, formatOption = "xls"):
    """# readIndata in to orig object
    input format could be either xls or txt:
    the defaulted type is xls
    if type is "txt" directly readin with del = '\t'
    if type is 'xls' read in data using prData class
    """
    if formatOption =='txt':
        orig = prData()
        rawset = open(dir_data, 'rU')
        firstline = rawset.readline()[:-1] #last bit is the change line symbol
        orig._fea = firstline.split('\t')
        for line in rawset:
            items = line[:-1].split('\t')
            tmp = dict()
            for i in range(len(items)):
                tmp[orig._fea[i]]=items[i]
            orig._data.append(tmp)
        return orig
    elif formatOption =='xls':
        orig = prData(dir_data)
        return orig




class Mytext (Text): #override the text class in NLTK so that the computed collocation could be retrived
	def collocations(self, num = 15, window_size = 2):
		if not ('_collocations' in self.__dict__ and self._num ==num and self._window_size == window_size):
			self._num = num
			self._window_size = window_size
			from nltk.corpus import stopwords
			ignored_words = stopwords.words('english')
			finder = nltk.collocations.BigramCollocationFinder.from_words(self.tokens, window_size)
			finder.apply_freq_filter(2)
			finder.apply_word_filter(lambda w: len(w) < 3 or w.lower() in ignored_words)
			bigram_measures = nltk.metrics.association.BigramAssocMeasures()
			self._collocations = finder.nbest(bigram_measures.likelihood_ratio, num)
		cp_list = [(w1, w2) for w1, w2 in self._collocations]
		return cp_list
	
	
	def getsimilar(self, word, num =20):
		"""
		@param word: The word used to seed the similarity search 
		@type word: C{str}  
		@param num: The number of words to generate (default=20) 
		@type num: C{int} 
		@seealso: L{ContextIndex.similar_words()}
		"""	
		if '_word_context_index' not in self.__dict__: 
			print 'Building word-context index...' 
			self._word_context_index = ContextIndex(self.tokens, 
		                                        filter=lambda x:x.isalpha(), 
		                                        key=lambda s:s.lower()) 
		#words = self._word_context_index.similar_words(word, num) 
		word = word.lower() 
		wci = self._word_context_index._word_to_contexts 
		if word in wci.conditions(): 
			contexts = set(wci[word]) 
			fd = FreqDist(w for w in wci.conditions() for c in wci[w] 
		              if c in contexts and not w == word) 
			words = fd.keys()[:num] #lists of words
			#print tokenwrap(words) 
			return words
		else: 
			print "No matches"
			return None
	
	
	def getfindall(self,regexp):
		"""
		find instance of the regular expression in the given text
		single token must be surrounded by angle brackets
		"""
		if "_token_searcher" not in self.__dict__:
			self._token_searcher = TokenSearcher(self)
		
		hits = self._token_searcher.findall(regexp)
		return hits
	

class topicReader():
	"""
	read the .ts file and store them as list of tuples (word, logratio)
	"""
	def __init__(self, dir_ts = None):
		self._wordlist = list()
		if dir_ts is not None:
			self.read(dir_ts)

	def read(self, dir_ts):
		input = open(dir_ts, 'r').read()
		lines = input.split('\n')
		for line in lines:
			if len(line)>0:
				[w, f] = line.split()
				self._wordlist.append((w, float(f)))
		self._wordlist = sorted(self._wordlist, key = operator.itemgetter(1),reverse=True) #descending

	def words(self):
		"""
		@return list of words with descending weight
		@rtype: list
		"""
		wordlist = map(operator.itemgetter(0), self._wordlist)
		return wordlist
	
	def getWordWeight(self):
		return self._wordlist

	
class prData():
#constructor  1) directory of xls data, 2) sheet index (starting from 0)
#delimiter, default = '<ITEMBREAK>'
	def __init__(self, dir_data=None, sheetIndex = None,stemmingOption=False, comName ='comment'):
		self._data = list() #list of dicts; each data point is treated as a dictionary
		self._fea = list() # map the feature name to the feature index in dataset
		self._comName = comName
		self._essayMapped = False
		if dir_data is not None:
			self.extractXLSsheet(dir_data, sheetIndex)
		self._delimiter = '<ITEMBREAK>'	
		#set stemming option
		if stemmingOption == False:
			self._stemoption =False
		else:
			self._stemoption = True		
		self.processComment()		
		#print "===== Read in orig data: ", len(self._data)	
	
	
	def extractXLSsheet(self, dir_xls, sheetIndex = None):
		book = xlrd.open_workbook(dir_xls)
		if sheetIndex is None:
			sheetIndex = 0
		
		sh = book.sheet_by_index(sheetIndex)
		for i in range(sh.ncols):
			self._fea.append(sh.cell_value(rowx=0, colx=i))
		#print "the imported featurelist is: ", self._fea
		for i in range(1,sh.nrows):
			ins = dict([(self._fea[k],sh.cell_value(rowx = i,colx = k)) for k in range(sh.ncols)])
			self._data.append(ins)
		#print "Read xls file: column ", sh.ncols,'; row ', sh.nrows
		

	
	def cleanUnicode(self, string):
		#u2022 is bullet sign, uf04c and uf04a are wired characters not neccessary to keep
		unicodes = {u'\uf0d8':'',  u'\u2014':'--',  u'\u201c':'\"', u'\u201D':'\"', u'\u2018':'\'', u'\u2019':'\'',  u'\u2026':'.',  u'\u2013':'-',  u'\u2022':'',  u'\uf04c':'', u'\uf04a':'',u'\xeb':'e', u'\xb0':' ',u'\xbe':'fraction three quarters',u'\uf010':'',u'\uf0e0':'',u'\xe9':'e', u'\uf0fc':'',u'\u201a':'a'}
		tmpString = string
		for code in unicodes.keys():
			p = re.compile(code)
			tmpString = p.sub(unicodes[code], tmpString)
		return self.filter_non_printable(tmpString)
		
	def filter_non_printable(self, str):
		return "".join([c for c in str if ord(c)<128])	
		
	#remove all NEWLINE character in the comment string
	#re.sub(pattern, repl, string[, count])
	def processComment(self):
		#find unicode list
		#remove empty comment (len(comment)==0)
		unicode_fea = []
		for fea in self._fea:
			if type(self._data[0][fea])==unicode:
				unicode_fea.append(fea)
		#print "unicode features in data: ", unicode_fea
		"""
		for i in range(len(self._data)):
			for fea in unicode_fea:
				self._data[i][fea] = self._data[i][fea].encode("utf-8")#.encode('ascii','ignore')#
				print self._data[i][fea]
		"""
		if self._comName in self._fea:
			p = '\n'
			toremovelist = []
			for i in range(len(self._data)):
				#print self._data[i]['ID']
				#print self._data[i][self._comName],type(self._data[i][self._comName])
				tmp  = self.cleanUnicode(self._data[i][self._comName])
				self._data[i][self._comName] = re.sub(p,' ',str(tmp))
				#print i, self._data[i][self._comName][:30]
				#remove empty comment instance
				if len(self._data[i][self._comName])==0:
					toremovelist.append(i)
			self._data = [self._data[i] for i in range(len(self._data)) if i not in toremovelist]					
		else:
			pass
	
		
#get the features in the fea_list of the specified data (default is the self.-data)
# return the dataset with only features in the fea_list
	def getdata_feaList(self, fea_list,data_1=None):
		fea_notfound = []
		if data_1 is None:
			fea_notfound = [key for key in fea_list if key not in self._fea] 
		if len(fea_notfound) >0:
			print "some feature doesn't exist: ", fea_notfound
		else:
			if data_1 is None:
				data = self._data
			else:
				data = data_1
				
			print "specified list: ",fea_list  #for check: requried fealist
			print "list in the data: ",data[0].keys()  # exisiting feature list in the required data set
			result = list()
			for ins in data:
				new_ins = dict([(key, ins[key]) for key in fea_list if key in self._fea])
				result.append(new_ins)
			
			return result
	
#filter by feature value, assume the data has all feature dimention
#return the subset of data
	def filter(self,feature, value, data=None, rev=False):		
		if data is None:
			data = self._data	
			
		if feature not in data[0].keys():
			print feature,"doesn't exist: "
			return None
		else:
			result = list()
			#index = self._fea[feature]
			for ins in data:
				if feature not in ins.keys():
					print data.index(ins), "feature doesn't exist!!!"
				else:
					if rev==False:
						if ins[feature] == value:
							result.append(ins)
					elif rev ==True:
						if ins[feature] != value:
							result.append(ins)
		
			return result

#filter from the raw data/structured data, 
# get the desired index list
# generate ready to use data by extracting corresponding instances directly from the provided arff file
	def GetIndexByFiltering(self,feature, value, data=None, rev=False):
		
		if data is None:
			data = self._data
			
		if feature not in self._fea:
			print feature,"doesn't exist: "
		else:
			result = list()  #result is the list of index
			#index = self._fea[feature]
			for ind in range(len(data)):
				ins = data[ind]
				if rev==False:
					if ins[feature] == value:
						result.append(ind)
				elif rev ==True:
					if ins[feature] != value:
						result.append(ind)
			
			return result




# delimiter is string type
# output instances with attributes seperated with specified delimiter
	def writeSeq(self, dir_output, data, fealist=None, delimiter=None):
		if delimiter is None:
			delim = self._delimiter
		else:
			delim = delimiter
		
		if fealist is None:
			fealist = data[0].keys()
				
		file_out = open(dir_output,'w')
		print len(data[0])
		for ins in data:
			output = [str(ins[k]) for k in fealist if k in ins.keys()]
			string = delim.join(output)
			if string[-1] == '\n':
				file_out.write(string)
			else:
				file_out.write(string+'\n')
		file_out.close()

	def writeTo(self, dir_output, data, delimiter=None,arff_head = None):
		if delimiter is None:
			delim = self._delimiter
		else:
			delim = delimiter
			
		file_out = open(dir_output,'w')
		print len(data[0])
		
		if arff_head is not None:
			#first write out the arffHead:
			ah = open(arff_head,'r').read()
			file_out.write(ah+'\n')
			
			
		for ins in data:
			print data.index(ins)
			output = [str(ins[k]) for k in ins.keys() if k in self._fea]
			string = delim.join(output)
			if string[-1] == '\n':
				file_out.write(string)
			else:
				file_out.write(string+'\n')
		file_out.close()

# for generate the arff data format, format the string type features with double quotes at each side.
	def writeFormat(self,dir_output,data,strlist,delimiter =None,arff_head = None):
			if delimiter is None:
				d =','
			else:
				d = delimiter
			ah = arff_head
			for ins in data:
				for key in strlist:
					ins[key] = '\"'+ins[key]+'\"'  #adding the quotes
			self.writeTo(dir_output,data,delimiter= d,arff_head = ah)  #call writeTo for general output purpose
	
	#for any given corpus writerNo. is in the middle
	#input: dir_essay, data, writerFeature, DocPattern_start, DocPattern_end
	def mapEssay2(self,dir_essay,DocPattern_start, DocPattern_end, data_1=None, writerFeature=None):
		self._essayMapped = True
		ef = 'essay' #name the new feature(directory of associated Essay)
		if data_1 is not None:
			data = data_1
		else:
			data = self._data
		
		if writerFeature is not None:
			wf = writerFeature
		else:
			wf = 'writerNo'
		
		doclist = os.listdir(dir_essay)
		self._fea.append(ef)

		notfoundessays = []
		for ins in data:	
			writerno = ins[wf]
			#print DocPattern_start + str(writerno) + DocPattern_end
			#p = re.compile(DocPattern_start + str(writerno)[:-2] + DocPattern_end) #@wenting changed on 05/13/2012
			p = re.compile(DocPattern_start + str(writerno) + DocPattern_end)
			filename = ''

			for doc in doclist:
				if p.search(doc) is not None:
					filename = doc
					break

			if filename != '': # found the corresponding essay
				"""
				essay = open(dir_essay+'/'+filename, 'r').read()
				e_tokens = nltk.wordpunct_tokenize(essay)
				ins[ef] = e_tokens
				"""
				
				ins[ef] = dir_essay+'/'+filename
			else:
				print "can't find the matched essay!", ins['ID'], writerno
				ins[ef] = None
				notfoundessays.append(ins[wf])
				
		#output error message if there is any missing essay			
		if len(notfoundessays) >0:
			print "There are missing essays: #=",len(list(set(notfoundessays)))
			print set(notfoundessays)			
		return data
	
	#for the given indexed comments
	def getEssayCollocation(self,data,index):
		if data[index]['essay'] is not None:
			essay_dir = data[index]['essay']
			text = open(essay_dir,'r').read()
			tokens = nltk.wordpunct_tokenize(text)
			tokens = [word.lower() for word in tokens]
		
			if self._stemoption ==True:
				st = PorterStemmer()
				tokens = [st.stem(t) for t in tokens]
			 
			t_bigram = set(Mytext(tokens).collocations())
			return t_bigram
		else:
			return set()
	
	"""
	added Nov 20th, 2010. for mapping essays collected from sword system.
	run sript_SWoRDRaw.py first to convertEssayName()
	"""
	def mapSWoRDEssay(self,dir_essay,DocPattern_start, DocPattern_end, data_1=None, writerFeature=None):
		self._essayMapped = True
		ef = 'essay' #name the new feature(directory of associated Essay)
		if data_1 is not None:
			data = data_1
		else:
			data = self._data

		if writerFeature is not None:
			wf = writerFeature
		else:
			wf = 'writerNo'

		doclist = os.listdir(dir_essay)
		self._fea.append(ef)

		notfoundessays = []
		for ins in data:	
			writerno = ins[wf]
			#print DocPattern_start + writerno + DocPattern_end
			p = re.compile(DocPattern_start + str(writerno) + DocPattern_end)
			filename = ''

			for doc in doclist:
				if p.search(doc) is not None:
					filename = doc
					break

			if filename != '': # found the corresponding essay
				"""
				essay = open(dir_essay+'/'+filename, 'r').read()
				e_tokens = nltk.wordpunct_tokenize(essay)
				ins[ef] = e_tokens
				"""

				ins[ef] = dir_essay+'/'+filename
			else:
				#print "can't find the matched essay!", ins['ID'], writerno
				ins[ef] = None
				notfoundessays.append(ins[wf])

		#output error message if there is any missing essay			
		if len(notfoundessays) >0:
			print "There are missing essays: #=",len(list(set(notfoundessays)))
			print set(notfoundessays)			
		return data	
	
	#extract unigram of all collocations of essays under the given directory
	def getDirCollocation_Word(self,directory):
		collocations = set()  #collocation items
		collo_uni = list()
		doclist = os.listdir(directory)
		for essay in doclist:
			dir_essay  = directory+'/'+essay
			etext = open(dir_essay,'r').read()
			tokens = nltk.wordpunct_tokenize(etext)
			tokens = [word.lower() for word in tokens]
			#stemming
			if self._stemoption ==True:
				st = PorterStemmer()
				tokens = [st.stem(t) for t in tokens]
			
			#extract the collocation for the given essay
			e_bigram = set(Mytext(tokens).collocations())
			collocations = collocations | e_bigram
			
		collocations = list(collocations)
		for (a, b) in collocations:
			collo_uni.append(a)
			collo_uni.append(b)
		
		collo_uni = list(set(collo_uni))		
		return collo_uni
	
					
	#extract bigram from the directory of txt files (above-average)	
	def getDomainBigram(self,directory=None):
		
		collocations = set()  #collocation items
		blists = list() #list of lists of bigrams
		
		#extract all bigrams into list of list of bigrams
		if directory is not None:
			doclist = os.listdir(directory)
			for essay in doclist:
				dir_essay  = directory+'/'+essay
				etext = open(dir_essay,'r').read()
				tokens = nltk.wordpunct_tokenize(etext)
				tokens = [word.lower() for word in tokens]
				#stemming
				if self._stemoption ==True:
					st = PorterStemmer()
					tokens = [st.stem(t) for t in tokens]
				
				#extract the collocation for the given essay
				e_bigram = set(Mytext(tokens).collocations())
				collocations = collocations | e_bigram
				btext = bigrams(tokens) #list of bigram
				blists.append(btext)
				
		else: # using the mapped essay to calcuate the candidate bigrams
			#need to call mapessay fuction first
			for ins in self._data:
				if ins['essay'] is not None:
					etext = open(ins['essay'],'r').read()
					tokens = nltk.wordpunct_tokenize(etext)
					tokens = [word.lower() for word in tokens]
					#stemming
					if self._stemoption ==True:
						st = PorterStemmer()
						tokens = [st.stem(t) for t in tokens]
				
					#extract the collocation for the given essay
					e_bigram = set(Mytext(tokens).collocations())
					collocations = collocations | e_bigram
					btext = bigrams(tokens)
					blists.append(btext)
		
		itemlist = list(collocations)
			
		bidf =[]
		value = 0
		total = len(blists)
		for item in itemlist: #here item is bigram:(a, b)
			for bilist in blists:
			#for test
				#print bilist
				if bilist.count(item): value=value+1
				else: pass
			if value != 0: 
				bidf.append((total/value, item))
			else: bidf.append((0.0, item))
			value = 0
			bidf = sorted(bidf, key = operator.itemgetter(0))
		
		ave = sum(map(operator.itemgetter(0), bidf)) / len(bidf)
		domainBlist = [j for (i, j) in bidf if i<ave]
		# turn bigram into list of tuple of words
		return domainBlist
	
	#(above-average)
	def getDomainBigram_Word(self, dir_essay=None): 
		dblist =self.getDomainBigram(directory=dir_essay)
		wlist = list()
		for (a, b) in dblist:
			wlist.append(a)
			wlist.append(b)
		wlist = list(set(wlist))	
		
		return wlist
	
	#(above-average)
	def getDomainUnigram(self, directory = None):		
		collocations = set()  #collocation items
		ewordlists = list() #list of lists of words
		
		#extract words from essays
		if directory is not None:
			doclist = os.listdir(directory)
			for essay in doclist:
				dir_essay  = directory+'/'+essay
				etext = open(dir_essay,'r').read()
				tokens = nltk.wordpunct_tokenize(etext)
				tokens = [word.lower() for word in tokens]
				#stemming
				if self._stemoption ==True:
					st = PorterStemmer()
					tokens = [st.stem(t) for t in tokens]
				
				#extract the collocation for the given essay
				e_bigram = set(Mytext(tokens).collocations())
				collocations = collocations | e_bigram
				ewordlists.append(tokens)
				
		else: # using the mapped essay to calcuate the candidate bigrams
			#need to call mapessay fuction first
			for ins in self._data:
				if ins['essay'] is not None:
					etext = open(ins['essay'],'r').read()
					tokens = nltk.wordpunct_tokenize(etext)
					tokens = [word.lower() for word in tokens]
					#stemming
					if self._stemoption ==True:
						st = PorterStemmer()
						tokens = [st.stem(t) for t in tokens]
				
					#extract the collocation for the given essay
					e_bigram = set(Mytext(tokens).collocations())
					collocations = collocations | e_bigram
					ewordlists.append(tokens)
		
		#get collection of all essays under the specified directory / associated essays
		collection_text = TextCollection(ewordlists)
		
		itemlist = list()
		for (a, b) in collocations:
			itemlist.append(a)
			itemlist.append(b)
			
		itemlist = list(set(itemlist))	
		
		word_idf = []
		for i in range(len(itemlist)):
			word_idf.append((collection_text.idf(itemlist[i]), itemlist[i]))	
		
		word_idf = sorted(word_idf, key = operator.itemgetter(0))
		ave = 0
		if len(word_idf)!=0:
			ave = sum(map(operator.itemgetter(0), word_idf)) / len(word_idf)
			
		wlist =  [j for (i, j) in word_idf if i<ave]				
		return wlist
	
	#(CNT of all Bigram collocation of the related essay)
	def AddTopicBigram(self, feaName,comName, data = None):
	#need mapping first
		if data is None:
			data =self._data
			
		for i in range(len(data)):	
			t_bigram = self.getEssayCollocation(data, i)
			#print t_bigram #tmp
			
			comment = data[i][comName]
			tokens = nltk.wordpunct_tokenize(comment)
			tokens = [word.lower() for word in tokens]
			#stemming
			if self._stemoption ==True:
				st = PorterStemmer()
				tokens = [st.stem(t) for t in tokens]
			comment_bigram = bigrams(tokens)
			#print comment_bigram  #tmp
			shared = [b for b in comment_bigram if b in t_bigram]
			#normalized
			data[i][feaName] = float(len(shared))/(len(tokens)+0.00001)
			#print data[i][feaName] #tmp
	
	#(CNT of all unigramOfBigram collocation of the related essay)		
	def AddTopicUnigram(self, feaName,comName, data = None):	
	#need mapping first
		if data is None:
			data =self._data
			
		for i in range(len(data)):	
			t_bigram = self.getEssayCollocation(data, i)
			
			t_uni = list()
			for (a, b) in t_bigram:
				t_uni.append(a)
				t_uni.append(b)
			t_uni = set(t_uni)
			
			comment = data[i][comName]
			tokens = nltk.wordpunct_tokenize(comment)
			tokens = [word.lower() for word in tokens]
		
			#stemming
			if self._stemoption ==True:
				st = PorterStemmer()
				tokens = [st.stem(t) for t in tokens]
				t_uni  = set([st.stem(t) for t in list(t_uni)])
			shared = [w for w in tokens if w in t_uni]
			#normalized
			data[i][feaName] = float(len(shared))/(len(tokens)+0.00001)

	
		
	#@Function: add one new feature to the given data
	# feaName: the name for the new extracted feature
	# values: TYPE: List
	# data: default = self._data
	def AddFeature(self,feaName,values,data = None):
		if data is None:
			data = self._data
			self._fea.append(feaName)
			
		for i in range(len(data)):
			data[i][feaName] = values[i]
	


	#@FUNCTION: merge data1 into data2
	#@TYPE data1 = data2 = list of dictionary
	#the sequence of data1 and data2 should be the same!!
	def Merge(self, data1,data2=None):
		if data2 is None: # add to self._data
			data2 = self._data
		
		#check the length
		if len(data1) !=len(data2):
			print 'data1 and data2 should correspond to the same sequence of data!'
			return None
		else:
			
			for i in range(len(data1)):
				newins = data1[i].items() + data2[i].items()
				data2[i] = dict(newins)
			
	
	def generateCorpus(self, dir_root, regPattern, dir_output):
		"""
		generate background corpus word counts from all docs of the given directory 
		
		@param dir_root: dir_root of the corpus docs
		@param regPattern:  '.*\.txt'
		@return: reader
		@rtype: plaintextCorpusReader
		"""	
		corpus = corpusReader(dir_root, regPattern)
		print corpus.words()
		fd= FreqDist(corpus.words()) 
		print fd.B()
		output = open(dir_output,'w')
		for w in fd.keys():
			newstring = w + " "+str(int(fd[w]))+'\n'
			output.write(newstring)
		output.close()
		return corpus
	
	"""
	draw histogram for the given list of samples
	@param samples: list of feature values
	@newTitle: the title of the generated histrogram (percentage as option)
	"""
		
	def readSWoRDratingFile(self, dir_rating, addToOrigdata = True, sheetIndex = None):
		book = xlrd.open_workbook(dir_rating)
		if sheetIndex is None:
			sheetIndex = 0
		sh = book.sheet_by_index(sheetIndex)
		features = list()
		ratings = list()
		if sh.ncols<=2:
			print "lack rating dimensions."
			return -1
		else:
			for i in range(2,sh.ncols):
				features.append(sh.cell_value(rowx=0, colx=i))
		print "rating dimensions are: ", features  #wrtier, reviewer, dimensions
		#readin ratings
		for i in range(1,sh.nrows):
			writer = sh.cell_value(rowx = i,colx = 0)
			reviewer = sh.cell_value(rowx = i,colx = 1)
			if writer!='' and reviewer!='':
				ins = dict([(features[k],sh.cell_value(rowx = i,colx = k+2)) for k in range(sh.ncols-2)])
				tmp = dict()
				tmp['writerNo'] = writer
				tmp['reviewerNo'] = reviewer
				tmp['rating'] = ins
				ratings.append(tmp)				
		if addToOrigdata ==True:
			#supplement the data orig with ratings from rating file
			for i in range(len(self._data)):
				wrt = self._data[i]['writerNo']
				rev = self._data[i]['reviewerNo']
				dim = self._data[i]['dim']
				rt= [ins for ins in ratings if ins['writerNo']==wrt and ins['reviewerNo']==rev][0]['rating']
				#print rt.keys(), dim, type(dim)
				if dim in rt.keys():
					self._data[i]['rating'] = int(float(rt[str(dim)]))
			#end supplement	
		return ratings
		
	def textPreprocessing(self):
		"""
		lowercase and remove stopwords
		"""
		ignored_words = nltk.corpus.stopwords.words('english')
		for i in range(len(self._data)):
			com = nltk.wordpunct_tokenize(self._data[i][self._comName])
			newc = []
			for j in range(len(com)):
				token = com[j].lower()
				#checking stopwords
				if token not in ignored_words:
					newc.append(token)
			self._data[i][self._comName] = " ".join(newc)
		#end
	


	def removeDomainWords(self, dir_domain, data = None,REP = None):
		"""
		remove the topicwords in instance comments
		all topic signatures are in lowercase
		@return: data
		"""
		if data is None:
			data = self._data
			
		topics = self.readTopicFromTS(dir_domain)
		if REP is None:
			rep = 'ddd'
		else:
			rep = REP
		for i in range(len(data)):
			com = nltk.wordpunct_tokenize(data[i][self._comName])
			for j in range(len(com)):
				if com[j].lower() in topics:
					com[j] = rep
			data[i][self._comName] = " ".join(com)
		return data
	

	def readTopicFromTS(self, dir_ts, withWeight = False):
		wordlist = list()
		input = open(dir_ts, 'r').read()
		lines = input.split('\n')
		for line in lines:
			if len(line)>0:
				[w, f] = line.split()
				wordlist.append((w, float(f)))
		wordlist = sorted(wordlist, key = operator.itemgetter(1),reverse=True) #descending
		if withWeight: #output word-weight pairs
			return wordlist
		else:
			return self.getWords(wordlist)

	def getWords(self, wordlist):
		"""
		@return list of words with descending weight
		@rtype: list
		"""
		words = map(operator.itemgetter(0), wordlist)
		return words
