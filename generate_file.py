#!/usr/bin/python

import os
import commands
import string
import re
import sys 
import gzip
import zipfile
import pickle
import random
from operator import itemgetter

global _tokenize_regexps
_tokenize_regexps = [
	(re.compile(r'\`'), r'\''),
	(re.compile(r'(\s|^)rt@[a-z0-9]+ '), r'-user- '),
	
	#(re.compile(r'&#39;'),r'\''),
	#(re.compile(r'&#96;'),r'\''),
	#(re.compile(r'rt'),r''),
	#(re.compile(r'&#34;'),r''),
	#(re.compile(r'&#63;'),r''),
	#(re.compile(r'&#34;'),r''),
	#(re.compile(r'&[a-z][a-z];'),r''),
	#(re.compile(r'&[a-z][a-z]'),r''),
	#(re.compile(r'\s#[a-z]+\s'),r' -topic- '),
	#(re.compile(r'\'\''), r'"'),
	#(re.compile(r'\`\`'), r'"'),

	# Combine dots separated by whitespace to be a single token:
	#(re.compile(r'\.\s\.\s\.'), r'...'),

	# fix %, $, &
	#(re.compile(r'(\d)%'), r' -percent- '),
	#(re.compile(r'\$(\.?\d)'), r' -money- '),
	#(re.compile(r'(\w)& (\w)'), r'\1&\2'),
	#(re.compile(r'(\w\w+)&(\w\w+)'), r'\1 & \2'),

	# Only separate comma if space follows:
	#(re.compile(r'(.)(,)(\s|$)'), r'\1 \2\3'),

	# Treat double-hyphen as one token:
	#(re.compile(r'([^-])(\-\-+)([^-])'), r'\1 \2 \3'),
	#(re.compile(r'(\s|^)(,)(?=(\S))'), r'\1\2 '),

	# Separate words from ellipses
	#(re.compile(r'([^\.]|^)(\.{2,})(.?)'), r'\1 \2 \3'),
	#(re.compile(r'(^|\s)(\.{2,})([^\.\s])'), r'\1\2 \3'),
	#(re.compile(r'([^\.\s])(\.{2,})($|\s)'), r'\1 \2\3'),

	# Separate punctuation (except period) from words:
	#(re.compile(r'(^|\s)(\')'), r'\1\2 '),
	#(re.compile(r'(?=[\(\"\`{\[:;&\#\*@])(.)'), r'\1 '),
    
	#(re.compile(r'(.)(?=[?!)\";}\]\*:@\'])'), r'\1 '),
	#(re.compile(r'(?=[\)}\]])(.)'), r'\1 '),
	#(re.compile(r'(.)(?=[({\[])'), r'\1 '),
	#(re.compile(r'((^|\s)\-)(?=[^\-])'), r'\1 '),
	]

def tokenize(s):
	"""
	Tokenize a string using the rule above
	"""
	global _tokenize_regexps
	for (regexp, repl) in _tokenize_regexps:
		s = regexp.sub(repl, s)
	return s

def save_pickle(data, path):
	sys.stderr.write("\nWriting lookup table file "+path+"\n")
	with open(path, 'wb') as f:
		pickle.dump(data, f)

def load_pickle(path):
	sys.stderr.write("\nLoading lookup table file "+path+"\n")
	with open(path, 'rb') as f:
		data = pickle.load(f)
		return data

def get_files(path, pattern):
	"""
	Recursively find all files rooted in <path> that match the regexp <pattern>
	"""
	L = []
    
	# base case: path is just a file
	if (re.match(pattern, os.path.basename(path)) != None) and os.path.isfile(path):
		L.append(path)
		return L

	# general case
	if not os.path.isdir(path):
		return L

	contents = os.listdir(path)
	for item in contents:
		item = path + item
		if (re.search(pattern, os.path.basename(item)) != None) and os.path.isfile(item):
			L.append(item)
		elif os.path.isdir(path):
			L.extend(get_files(item + '/', pattern))

	return L


if __name__ == '__main__':
	#debug()
	files = get_files("./",".*")
	for file in files:
		if file.find("py") >= 0:continue
		if file.find(".bat") >= 0:continue
		lines = open(file).read().splitlines() 
		f = open("./new/"+file,'w')
		p = 0
		for line in lines:
			line = line.replace("''","\"")
			line = line.replace("``","\"")
			if line.lower() == "<p>":
				p=1
				f.write(line+"\n")
				tmp=""
			if line.lower() == "</p>":
				p=0
				f.write(tmp+"\n")
				tmp=""
			if p == 1 and line.lower() != "<p>":
				tmp = tmp+" "+line
			elif p == 0:
				f.write(line+"\n")
		f.close	
