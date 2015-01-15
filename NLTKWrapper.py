#!/usr/bin/env python
#file: NLTKWrapper.py

import nltk
import nltk.data
import porter

punctuations = ".?!:;-()[]'\"/,"

def splitSentence(paragraph):
    sentences = []
    paragraph = paragraph.strip()
    paragraph = paragraph.replace("\r\n", "\n")
    firstSplitsentences = paragraph.split("\n")
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    
    for s in firstSplitsentences:
        secondspits = tokenizer.tokenize(s)
        sentences = sentences + secondspits
    return sentences
  
def wordtokenizer(s, punct = True):
    if punct:
        return nltk.wordpunct_tokenize(s)
    else:
        return nltk.word_tokenize(s)

def getNgram(sentence, n, punct = True):
    #n is the number of grams, such as 1 means unigram
    ngrams = []
    
    #tokens = summary.split()
    tokens = wordtokenizer(sentence, punct)
    N = len(tokens)
    for i in range(N):
        if i+n > N: continue
        ngram = tokens[i:i+n]
        ngrams.append(" ".join(ngram))
    return ngrams

def getNgramTokened(word, n, tag = None):
    #n is the number of grams, such as 1 means unigram
    ngram_tags = []
    ngram_words = []
    
    if tag != None:
        assert(len(tag) == len(word))
    
    #tokens = summary.split()
    N = len(word)
    for i in range(N):
        if i+n > N: continue
        
        if tag != None:
            ngram_tag = tag[i:i+n]
            ngram_tags.append(" ".join(ngram_tag))
            
        ngram_word = word[i:i+n]
        ngram_words.append(" ".join(ngram_word))
    
    if tag != None:
        return ngram_words, ngram_tags 
    
    return ngram_words

def getWordList(file):
    f = open(file,'r')
    lines = f.readlines()
    f.close()
    
    words = []
    for line in lines:
        tokens = wordtokenizer(line)
        words = words + tokens
    
    return words

if __name__ == '__main__':
    print splitSentence("[1] I love you. [2] Sent 2. [3] sentence 3")