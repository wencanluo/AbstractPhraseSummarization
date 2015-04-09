import fio
import json
import sys
import porter
import NLTKWrapper
import os
import numpy
import NumpyWrapper
import Survey
import ILP_baseline as ILP
import ILP_SVD

from ILP_baseline import stopwords

phraseext = ".key" #a list
studentext = ".keys.source" #json
countext = ".dict"  #a dictionary
featureext = ".f"

titledir = "E:/Dropbox/reflection project_LRDC/250 Sp11 CLIC All Lecs .2G/titles/"

def getNgram(prefix, ngram):
    phrases, bigrams, PhraseBigram = ILP.getPhraseBigram(prefix + phraseext, Ngram=ngram)
    
    #get weight of bigrams {bigram:weigth}
    BigramTheta = ILP.getBigramWeight_TF(PhraseBigram, phrases, prefix + countext) # return a dictionary
    
    dict = {}
    for bigram, tf in BigramTheta.items():
        bigramname = bigrams[bigram]
        dict[bigramname] = tf
        
    return dict

def extract_TF(prefix, ngram):
    phrases, bigrams, PhraseBigram = ILP.getPhraseBigram(prefix + phraseext, Ngram=ngram)
    
    #get weight of bigrams {bigram:weigth}
    BigramTheta = ILP.getBigramWeight_TF(PhraseBigram, phrases, prefix + countext) # return a dictionary
    
    dict = {}
    for bigram, tf in BigramTheta.items():
        bigramname = bigrams[bigram]
        dict[bigramname] = tf
        
    return dict

def extract_title(prefix, ngram, titlefile):
    phrases, bigrams, PhraseBigram = ILP.getPhraseBigram(prefix + phraseext, Ngram=ngram)
    titles = Survey.getTitle(titlefile)
    
    titledict = {} #the ngram vocabulary in the titles
    for title in titles:
        #get stemming
        phrase = porter.getStemming(title.lower())
        tokens = phrase.lower().split()
        #tokens = list(gensim.utils.tokenize(phrase, lower=True, errors='ignore'))
        
        #get bigrams
        ngrams = []
        for n in ngram:
            grams = ILP.getNgramTokenized(tokens, n, NoStopWords=True)
            ngrams = ngrams + grams
            
        for word in ngrams:
            titledict[word.lower()] = True
        
    dict = {}
    for bigram in bigrams:
        bigramname = bigrams[bigram]
        dict[bigramname] = 1 if bigramname.lower() in titledict else 0
        
    return dict

def extract_one(prefix, ngram):
    phrases, bigrams, PhraseBigram = ILP.getPhraseBigram(prefix + phraseext, Ngram=ngram)
   
    #get weight of bigrams {bigram:weigth}
    BigramTheta = ILP.getBigramWeight_TF(PhraseBigram, phrases, prefix + countext) # return a dictionary
    
    dict = {}
    for bigram, tf in BigramTheta.items():
        bigramname = bigrams[bigram]
        dict[bigramname] = 1
        
    return dict

def get_nonstop_ratio(bigram):
    words = bigram.split()
    
    stop_wc = 0.0
    for word in words:
        if word in stopwords:
            stop_wc = stop_wc + 1.0
    
    return 1-stop_wc/len(words)

def extract_nonstop_ratio(prefix, ngram):
    phrases, bigrams, PhraseBigram = ILP.getPhraseBigram(prefix + phraseext, Ngram=ngram)
    
    dict = {}
    for bigram in bigrams:
        bigramname = bigrams[bigram]
        dict[bigramname] = get_nonstop_ratio(bigramname)
    return dict

def extract_ngram_length(prefix, ngram):
    phrases, bigrams, PhraseBigram = ILP.getPhraseBigram(prefix + phraseext, Ngram=ngram)
    
    dict = {}
    for bigram in bigrams:
        bigramname = bigrams[bigram]
        dict[bigramname] = len(bigramname.split())
    return dict
        
def add_feature_set(todict, fromdict):
    for k, v in fromdict.items():
        if k not in todict:
            todict[k] = []
        if type(v) == list:
            todict[k] = todict[k] + v
        else:
            todict[k].append(v)
    return todict
           
def extract_single(prefix, ngram, output, titlefile=None):
    data = {}
    
    tf_dict = extract_TF(prefix, ngram)
    one_dict = extract_one(prefix, ngram)
    stop_ratio_dict = extract_nonstop_ratio(prefix, ngram)
    ngram_length_dict = extract_ngram_length(prefix, ngram)
    
    if titlefile != None:
        title_dict = extract_title(prefix, ngram, titlefile)
    else:
        title_dict = {}
    
    #data = add_feature_set(data, one_dict)
    data = add_feature_set(data, tf_dict)
    #data = add_feature_set(data, stop_ratio_dict)
    #data = add_feature_set(data, ngram_length_dict)
    #data = add_feature_set(data, title_dict)
    
    with open(output, 'w') as outfile:
        json.dump(data, outfile, indent=2)

def extact(ilpdir, np, ngram):
    sheets = range(0,12)
    
    for i, sheet in enumerate(sheets):
        week = i + 1
        dir = ilpdir + str(week) + '/'
        
        for type in ['POI', 'MP', 'LP']:
            prefix = dir + type
            feature_file = prefix + featureext
            
            prefix = prefix + '.' + np
            print feature_file 
            
            titlefile = titledir + str(week) + '.TXT'
            
            extract_single(prefix, ngram, feature_file, titlefile)
                    
if __name__ == '__main__':   
    ilpdir = "../../data/ILP_Sentence_Supervised_FeatureWeighting/"
    
    extact(ilpdir, np = 'sentence', ngram=[1,2])
    