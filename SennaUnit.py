## @package SennaUnit
# This script is used to store semantic role labeling for a sentence (a semantic role labeling  toolkit, see http://ml.nec-labs.com/senna/)
# @author Wencan Luo (wencanluo.cn@gmail.com)
#
class SennaWord:
	"""
	@function: Word Object for Senna
		@attribute token: string, word lexicon
		@attribute pos: string, part of speech
		@attribute chk: string, chunk tag
		@attribute srl_verb : Semantic Role Verb
		@attribute ner: string, Name entity tag
		@attribute srl_role: string, Semantic Role Labeling tag
		@attribute psg: string, syntactic parsing tag
		@attribute slot: slot value tag
	"""
	def __init__(self):
		self.token = ""
		self.pos = ""	#Part of Speech (POS)
		self.chk = ""	#Chunking (CHK)
		self.ner = ""	#Name Entity Recognition (NER)
		self.srl_verb = ""	#Semantic Role Verb
		self.srl_role = []	#Semantic Role Labeling (SRL)
		self.psg = ""	#Syntactic Parsing (PSG)
		self.slot = ""	#slot value
		
	def __init__(self, row):
		#print len(row), row
		self.token = row[0]
		self.pos = row[1]
		self.chk = row[2]
		self.ner = row[3]
		self.srl_verb = row[4]
		self.srl_role = []
		self.psg = row[len(row)-1]
		
		for i in range(5, len(row)-1):
			self.srl_role.append(row[i])
	
	def hasSRLRole(self):
		if len(self.srl_role) > 0: return True
		return False
	
	def getSRLRole(self):
		if self.hasSRLRole():
			return self.srl_role[0]
		else:
			return "-"
				
	def __str__(self):
		s = ""
		s = s + self.token + "\t"
		s = s + self.pos + "\t"
		s = s + self.chk + "\t"
		s = s + self.ner + "\t"
		s = s + self.srl_verb + "\t"
		for role in self.srl_role:
			s = s + role + "\t"
		s = s + self.psg
		s = s + "\r\n"
		return s
	
class SennaSentence:
	"""
	@function: Sentence Object for Senna
		@attribute words: list, a list of SennaWord
		@attribute label: string, the topic of the sentence
		@attribute count: int, the frequency of the sentence
		@attribute slot: string, the slot tag of the sentence
	"""
	def __init__(self):
		self.words = []
		self.label = ""
		self.count = 0
		self.slot = ""
		
		self.NounTag = {'NN':1, 'NNS':1, 'NNP':1, 'NNPS':1}
		self.AdjTag = {'JJ':1, "JJR":1, "JJS":1}
		
	def __init__(self, grid):
		"""
		Create a sentence from the grid produced by SENNA
		"""
		
		self.NounTag = {'NN':1, 'NNS':1, 'NNP':1, 'NNPS':1}
		self.AdjTag = {'JJ':1, "JJR":1, "JJS":1}
		
		self.words = []
		self.label = ""
		#print "grid len = ", len(grid)
		for row in grid:
			#print "row len = ", len(row)
			self.words.append( SennaWord(row) )

	def __str__(self):
		s = ""
		for word in self.words:
			s = s + str(word)
		s = s + "\r\n"
		
		return s
	
	def setLabel(self, label):
		"""
		@function: set the topic of the sentence
		@param label: string, the topic string 
		"""
		self.label = label
	
	def getLabel(self):
		return self.label
	
	def setCount(self, count):
		self.count = count
	
	def getCount(self):
		return self.count

	def getWordsAsList(self):
		"""
		@function: get the word in SennaWord form
		"""
		return self.words 
			
	def getWords(self):
		words = ""
		for word in self.words:
			words = words + word.token + " "
		return words.strip()
	
	def getNPrases(self):
		NP = []
		
		tmp = []
		start = False
		for word in self.words:
			if word.chk == 'S-NP': #single word NP phrase
				NP.append(word.token)
				tmp = []
				start = False
				
			elif word.chk == 'B-NP': # begin
				start = True
				tmp = []
				tmp.append(word.token)
			elif word.chk == 'I-NP': # inside
				assert(start)
				tmp.append(word.token)
			elif word.chk == 'E-NP': # end
				assert(start)
				tmp.append(word.token)
				NP.append(" ".join(tmp))
				tmp = []
				start = False	
			else:
				assert(~start)
			
		return NP
	
	def getPrases(self):
		tags = []
		words = []
		
		tmp = []
		for word in self.words:
			if word.chk.startswith('S-'): #single word phrase
				tag = word.chk[2:]
				
				tags.append(tag)
				words.append(word.token)
				
				tmp = []
				
			elif word.chk.startswith('B-'): # begin
				tag = word.chk[2:]
				tmp = []
				tmp.append(word.token)
			elif word.chk.startswith('I-'): # inside
				tag = word.chk[2:]
				tmp.append(word.token)
			elif word.chk.startswith('E-'): # end
				tag = word.chk[2:]
				tmp.append(word.token)
				
				tags.append(tag)
				words.append(" ".join(tmp))
				tmp = []
			else: #O
				pass
			
		return words, tags
	
	def getSyntaxNP(self):
		NP = []
		
		#print self.getWords()
		
		stack = []
		tmp = []
		
		hasNP = False
		for word in self.words:
			Added = False
			
			for i, ch in enumerate(word.psg):
				if ch == '(':
					if not hasNP: continue
					stack.append(ch)
				elif ch == ')':
					if not hasNP: continue
					
					stack.pop()
					if len(stack) == 0: #empty, the NP is done
						if not Added: 
							tmp.append(word.token)
							Added = True
						NP.append(" ".join(tmp))
						tmp = []
						hasNP = False
				else:
					if hasNP: 
						if not Added: #add the word only once
							tmp.append(word.token)
							Added = True
						continue
					
					if word.psg[i-1:].startswith('(NP'):
						hasNP = True
						stack.append('(')
						tmp.append(word.token)
						Added = True
		
		return NP
	
	def getPhrasewithPos(self):#SyntaxNP
		NP = []
		
		stack = []
		tmp = []
		
		hasNP = False
		for word in self.words:
			Added = False
			
			for i, ch in enumerate(word.psg):
				if ch == '(':
					if not hasNP: continue
					stack.append(ch)
				elif ch == ')':
					if not hasNP: continue
					
					stack.pop()
					if len(stack) == 0: #empty, the NP is done
						if not Added: 
							tmp.append(word.token.lower() + "/" + word.pos)
							Added = True
						NP.append(" ".join(tmp))
						tmp = []
						hasNP = False
				else:
					if hasNP: 
						if not Added: #add the word only once
							tmp.append(word.token.lower() + "/" + word.pos)
							Added = True
						continue
					
					if word.psg[i-1:].startswith('(NP'):
						hasNP = True
						stack.append('(')
						tmp.append(word.token.lower() + "/" + word.pos)
						Added = True
		
		return NP
	
	def getPhrasPairwithPos(self):#SyntaxNP
		# return [NP: NP+Pos]
		NP = []
		
		stack = []
		tmp = []
		tmpNP = []
		
		hasNP = False
		for word in self.words:
			Added = False
			
			for i, ch in enumerate(word.psg):
				if ch == '(':
					if not hasNP: continue
					stack.append(ch)
				elif ch == ')':
					if not hasNP: continue
					
					stack.pop()
					if len(stack) == 0: #empty, the NP is done
						if not Added: 
							tmpNP.append(word.token)
							tmp.append(word.token.lower() + "/" + word.pos)
							Added = True
						NP.append((" ".join(tmpNP), " ".join(tmp)))
						tmp = []
						tmpNP = []
						hasNP = False
				else:
					if hasNP: 
						if not Added: #add the word only once
							tmpNP.append(word.token)
							tmp.append(word.token.lower() + "/" + word.pos)
							Added = True
						continue
					
					if word.psg[i-1:].startswith('(NP'):
						hasNP = True
						stack.append('(')
						tmpNP.append(word.token)
						tmp.append(word.token.lower() + "/" + word.pos)
						Added = True
		
		return NP
	
	def getAdjNounPrases(self):
		AdjNoun = []
		
		tmp = []
		start = False
		
		dictNoun = {}
		dictAdj = {}
		
		#get noun and adjective dict
		for i,word in enumerate(self.words):
			if word.pos in self.NounTag:
				if word.token not in dictNoun: #two words might be the same
					dictNoun[word.token]  = []
				dictNoun[word.token].append(i)
		
			if word.pos in self.AdjTag:
				if word.token not in dictAdj:
					dictAdj[word.token]  = []
				dictAdj[word.token].append(i)
		
		for noun in dictNoun:
			for p in dictNoun[noun]: #for every position
				nearestAdj = None
				nearestp = -1
				
				for adj in dictAdj:
					for p2 in dictAdj[adj]:
						if nearestAdj == None:
							nearestAdj = adj
							nearestp = p2
						else:
							if abs(p2-p) < abs(nearestp - p):
								nearestAdj = adj
								nearestp = p2
				
				if nearestAdj != None: #find one, add to the candidate
					AdjNoun.append(nearestAdj + " " + noun)
				else:
					AdjNoun.append(noun)
			
		return AdjNoun
	
	def getPos(self):
		"""
		@function: get the pos string in SennaWord form
		"""
		pos = ""
		for word in self.words:
			pos = pos + word.pos + " "
		return pos.strip()
	
	def getWordPos(self, tolower=False):
		"""
		@function: get the pos string in SennaWord form
		"""
		res = ""
		for word in self.words:
			if tolower:
				res = res + word.token.lower() + "/" +  word.pos + " "
			else:
				res = res + word.token + "/" +  word.pos + " "
		return res.strip()
	
	def getNer(self):
		"""
		@function: get the NER string in SennaWord form
		"""
		ner = ""
		for word in self.words:
			ner = ner + word.ner + " "
		return ner.strip()
	
	def getChunk(self):
		"""
		@function: get the Chunk string in SennaWord form
		"""
		chk = ""
		for word in self.words:
			chk = chk + word.chk + " "
		return chk.strip()
	
	def getSRLVerb(self):
		"""
		@function: get the Chunk string in SennaWord form
		"""
		srl_verb = ""
		for word in self.words:
			srl_verb = srl_verb + word.srl_verb + " "
		return srl_verb.strip()		
	
	def hasSRLRole(self):
		if len(self.words) < 1: return False
		if len(self.words[0].srl_role) >= 1: return True
		return False
			
	def getSRLRole(self):
		"""
		@function: get the first SRL role string in SennaWord form
		"""
		srl_role = ""
		for word in self.words:
			if word.hasSRLRole():
				srl_role = srl_role + word.srl_role[0] + " "
			else:
				srl_role = srl_role + "-" + " "
		return srl_role.strip()
	
	def getWordPlusPos(self):
		wordPlusPos = ""
		for word in self.words:
			wordPlusPos = wordPlusPos + word.token + "_" + word.pos + " "
		return wordPlusPos.strip()
	
	def setSlot(self, slot):
		self.slot = slot
		
		slots = slot.split()
		if(len(slots) != len(self.words)):
			print slots
			print self.words
			print self.getWords()
		
		for i in range(len(self.words)):
			#if slots[i][0] != '/':
			#	print "Error", slots, self.getWords()
			self.words[i].slot = slots[i][1:]
	
	def getSlot(self):
		return self.slot
		
if __name__ == "__main__":
	s = SennaSentence()
	print s