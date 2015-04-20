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
from collections import defaultdict

from feat_vec import FeatureVector 

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
        
        feat_vec = FeatureVector()
        if tf == 0: feat_vec['term_freq_0'] = 1.0
        if tf >= 1: feat_vec['term_freq_1'] = 1.0   
        if tf >= 2: feat_vec['term_freq_2'] = 1.0  
        if tf >= 5: feat_vec['term_freq_5'] = 1.0   
        if tf >= 10: feat_vec['term_freq_10'] = 1.0

        dict[bigramname] = feat_vec
        
    return dict

def extract_TF_Rank(prefix, ngram, topK=10):
    phrases, bigrams, PhraseBigram = ILP.getPhraseBigram(prefix + phraseext, Ngram=ngram)
    
    #get weight of bigrams {bigram:weigth}
    BigramTheta = ILP.getBigramWeight_TF(PhraseBigram, phrases, prefix + countext) # return a dictionary
    
    keys = sorted(BigramTheta, key=BigramTheta.get, reverse=True)
    
    dict = {}
    for i, bigram in enumerate(keys):
        bigramname = bigrams[bigram]
        
        feat_vec = FeatureVector()
        
        if i< topK: feat_vec['tf_rank_' + str(i)] = 1.0
        
        dict[bigramname] = feat_vec
        
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
        
        feat_vec = FeatureVector()
        if bigramname.lower() in titledict:
            feat_vec['in_title'] = 1.0
        
        dict[bigramname] = feat_vec
        
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
        
        feat_vec = FeatureVector()
        r = get_nonstop_ratio(bigramname)
        
        if r <= 0.5: feat_vec['non_stop_ratio_0.5'] = 1.0
        if r > 0.5: feat_vec['non_stop_ratio_1'] = 1.0 
        
        dict[bigramname] = feat_vec
        
    return dict

def get_word_count_dis(input, CountFile):
    lines = fio.ReadFile(input)
    phrases = [line.strip() for line in lines]
    
    CountDict = fio.LoadDict(CountFile, 'float')
    
    dict = defaultdict(float)
    for phrase in phrases:
        fequency = CountDict[phrase]
        
        #get stemming
        phrase = porter.getStemming(phrase)
        tokens = phrase.lower().split()
        
        words = ILP.getNgramTokenized(tokens, 1, NoStopWords=True)
        
        for word in words:
            dict[word] += fequency
    return dict
    
def extract_frequency_of_words(prefix, ngram):
    dict = {}
    
    #get all the word frequency
    wf_dict = get_word_count_dis(prefix + phraseext, prefix + countext)
    phrases, bigrams, PhraseBigram = ILP.getPhraseBigram(prefix + phraseext, Ngram=ngram)
    
    for bigram in bigrams:
        if bigram == 'all':
            print "all"
        bigramname = bigrams[bigram]
        
        feat_vec = FeatureVector()
        
        words = bigramname.split()
        if len(words) == 2:
            n1 = wf_dict[words[0]]
            n2 = wf_dict[words[1]]
            
            if n1 == 0:feat_vec['freq_firstw_0'] = 1.0 
            if n1 >= 1: feat_vec['freq_firstw_1'] = 1.0 
            if n1 >= 2: feat_vec['freq_firstw_2'] = 1.0 
            if n1 >= 3: feat_vec['freq_firstw_3'] = 1.0 
            if n1 >= 4: feat_vec['freq_firstw_4'] = 1.0 
            if n1 >= 5: feat_vec['freq_firstw_5'] = 1.0 
            if n1 >= 10: feat_vec['freq_firstw_10'] = 1.0 
            
            if n2 == 0: feat_vec['freq_secondw_0'] = 1.0 
            if n2 >= 1: feat_vec['freq_secondw_1'] = 1.0 
            if n2 >= 2: feat_vec['freq_secondw_2'] = 1.0 
            if n2 >= 3: feat_vec['freq_secondw_3'] = 1.0 
            if n2 >= 4: feat_vec['freq_secondw_4'] = 1.0 
            if n2 >= 5: feat_vec['freq_secondw_5'] = 1.0 
            if n2 >= 10: feat_vec['freq_secondw_10'] = 1.0 
        else:#unigram
            n1 = wf_dict[words[0]]
            
            if n1 == 0: feat_vec['freq_firstw_0'] = 1.0 
            if n1 >= 1: feat_vec['freq_firstw_1'] = 1.0 
            if n1 >= 2: feat_vec['freq_firstw_2'] = 1.0 
            if n1 >= 3: feat_vec['freq_firstw_3'] = 1.0 
            if n1 >= 4: feat_vec['freq_firstw_4'] = 1.0 
            if n1 >= 5: feat_vec['freq_firstw_5'] = 1.0 
            if n1 >= 10: feat_vec['freq_firstw_10'] = 1.0 
            
        dict[bigramname] = feat_vec
        
    return dict

def extract_ngram_length(prefix, ngram):
    phrases, bigrams, PhraseBigram = ILP.getPhraseBigram(prefix + phraseext, Ngram=ngram)
    
    dict = {}
    for bigram in bigrams:
        bigramname = bigrams[bigram]
        
        feat_vec = FeatureVector()
        if len(bigramname.split()) == 1: feat_vec['ngram_1'] = 1.0 
        if len(bigramname.split()) == 2: feat_vec['ngram_2'] = 1.0 
        
        dict[bigramname] = feat_vec
    return dict
        
def add_feature_set(todict, fromdict):
    for k, v in fromdict.items():
        if k not in todict:
            todict[k] = {}
        if type(v) == dict or type(v) == FeatureVector:
            for name, val in v.iteritems():
                todict[k][name] = val
    return todict
           
def extract_single(prefix, ngram, output, titlefile=None):
    data = {}
    
    tf_dict = extract_TF(prefix, ngram)
    tf_rank_dict = extract_TF_Rank(prefix, ngram)  
    one_dict = extract_one(prefix, ngram)  
    stop_ratio_dict = extract_nonstop_ratio(prefix, ngram)    
    ngram_length_dict = extract_ngram_length(prefix, ngram)
    frequency_of_words_dict = extract_frequency_of_words(prefix, ngram)
    if "all" in frequency_of_words_dict:
        print 'frequency_of_words_dict'
        
    if titlefile != None:
        title_dict = extract_title(prefix, ngram, titlefile)
    else:
        title_dict = {}
        
    #data = add_feature_set(data, one_dict)
    data = add_feature_set(data, tf_dict)
    data = add_feature_set(data, tf_rank_dict)
    data = add_feature_set(data, stop_ratio_dict)
    data = add_feature_set(data, ngram_length_dict)
    data = add_feature_set(data, title_dict)
    data = add_feature_set(data, frequency_of_words_dict)
    
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
    ilpdir = "../../data/ILP_Sentence_Supervised_FeatureWeightingMC/"
    
    extact(ilpdir, np = 'sentence', ngram=[1,2])
    