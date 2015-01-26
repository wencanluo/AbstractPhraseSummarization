## @package SennaParser
# This script is used to parse the file given by SENNA (a semantic role labeling  toolkit, see http://ml.nec-labs.com/senna/)
# @author Wencan Luo (wencanluo.cn@gmail.com)
#Usage:
#		python SennaParser.py filename
#
# Usage Example:
#python SennaParser.py ../data/outputFromSenna.txt
#

import sys
import fio
from SennaUnit import *

def SennaParseWithCountDict(filename):
	"""
	@function: Parse the file and return a list of sentence with index.
	@param filename: string, the filename of the sennafile, the sennafile is an output file given by SENNA
	@return: <list, dict>, the dict stores for the start line for each sentence
	"""
	lines = fio.ReadFile(filename)
	print "nLine=", len(lines)
	sys.stdout.flush()

	CountDict = {}
	
	nCount = 0
	nLast = -1
	for i in range(len(lines)):
		line = lines[i]
		row = []
		line = line.strip()
		if len(line) == 0: #the last sentence is finished
			CountDict[nCount] = nLast+1
			nLast = i
			nCount = nCount + 1
	print "nCount=", nCount
	sys.stdout.flush()
	
	#for s in sentences:
	#	print s
	return lines, CountDict

def SennaParse(filename):
	"""
	@function: Parse the file and return a list of sentence. Each sentence is a SennaSentence
	@param filename: string, the filename of the sennafile, the sennafile is an output file given by SENNA
	@return: list, Each item is a SennaSentence
	"""
	lines = fio.ReadFile(filename)
	#print "nLine=", len(lines)
	sys.stdout.flush()

	nCount = 0
	for line in lines:
		row = []
		line = line.strip()
		if len(line) == 0: #the last sentence is finished
			nCount = nCount + 1
	#print "nCount=", nCount
	sys.stdout.flush()
	
	sentences = [None]*nCount
	nCount = 0
	
	tm = []
	for line in lines:
		row = []
		line = line.strip()
		if len(line) == 0: #the last sentence is finished
			sentences[nCount] = SennaSentence(tm)
			nCount = nCount + 1
			tm = []
			continue
		
		for num in line.split("\t"):
			row.append(num.strip())
		tm.append(row)
	
	#for s in sentences:
	#	print s
	return sentences
		
if __name__ == "__main__":
	if len(sys.argv) != 2:
		#print "Incorrect Input. Usage: python SennaParser.py filename"
		#sys.exit()
		filename = '../data/senna/senna.2.MP.output'
	else:
		filename = sys.argv[1]
		
	sentences = SennaParse(filename)
	print len(sentences)
	
	print sentences[0]
	#lines, dict = SennaParseWithCountDict('H:/svn/nlu/20_Data/TextData/outputFromSenna.txt')
	#fio.PrintDict(dict)