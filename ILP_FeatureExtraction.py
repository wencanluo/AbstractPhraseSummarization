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
import util
from feat_vec import FeatureVector 

from ILP_baseline import stopwords, stopwordswithpunctuations

import get_ngram_NP
import get_ngram_pos
import get_ngram_tfidf

nonwords = [line.lower().strip() for line in fio.ReadFile("../../data/wordsEn.txt")]
nonwords = [porter.getStemming(w) for w in nonwords]

phraseext = ".key" #a list
studentext = ".keys.source" #json
countext = ".dict"  #a dictionary
featureext = ".f"
tfidfext = ".tfidf"
posext = '.pos'
inNPext = '.inNP'

titledir = "E:/Dropbox/reflection project_LRDC/250 Sp11 CLIC All Lecs .2G/titles/"

prompts = {'POI':'Describe what you found most interesting in today\'s class.',
           'MP':'Describe what was confusing or needed more detail.',
           'LP':'Describe what you learned about how you learn.'}

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
        #if tf == 0: feat_vec['term_freq=0'] = 1.0 #remove
        if tf > 1: feat_vec['term_freq>=1'] = 1.0   
        if tf > 2: feat_vec['term_freq>=2'] = 1.0  
        if tf > 5: feat_vec['term_freq>=5'] = 1.0
        
        #double
        #if tf <= 1: feat_vec['term_freq>=1'] = 1.0   
        if tf <= 2: feat_vec['term_freq<=2'] = 1.0  
        if tf <= 5: feat_vec['term_freq<=5'] = 1.0
           
#         if tf >= 10: feat_vec['term_freq>=10'] = 1.0 #remove
        
#         if tf != 0: feat_vec['term_freq!=0'] = 1.0 #remove
#         if tf < 1: feat_vec['term_freq<1'] = 1.0   
#         if tf < 2: feat_vec['term_freq<2'] = 1.0  
#         if tf < 5: feat_vec['term_freq<5'] = 1.0   
#         if tf < 10: feat_vec['term_freq<10'] = 1.0

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
        
        if i<5: feat_vec['tf_rank<5'] = 1.0
        if i<10: feat_vec['tf_rank<10'] = 1.0
        
#         for k in range(i, topK):
#             feat_vec['tf_rank<=' + str(k)] = 1.0
#         
#         minK = min(topK, i)
#         for k in range(0, minK):
#             feat_vec['tf_rank>' + str(k)] = 1.0
        
        dict[bigramname] = feat_vec
        
    return dict

def extract_TFIDF(prefix, ngram):
    BigramTFIDF = fio.LoadDict(prefix + tfidfext, float)
    
    dict = {}
    for bigram, tfidf in BigramTFIDF.items():
        
        feat_vec = FeatureVector()
        #if tfidf >= 0.5: feat_vec['tfidf>=0.5'] = 1.0
        #if tfidf >= 0.1: feat_vec['tfidf>=0.1'] = 1.0
        if tfidf > 0.05: feat_vec['tfidf>0.05'] = 1.0
        #if tfidf >= 0.04: feat_vec['tfidf>=0.04'] = 1.0
        #if tfidf >= 0.03: feat_vec['tfidf>=0.03'] = 1.0
        if tfidf > 0.02: feat_vec['tfidf>0.02'] = 1.0
        if tfidf > 0.01: feat_vec['tfidf>0.01'] = 1.0
        #if tfidf >= 0.005: feat_vec['tfidf>=0.005'] = 1.0
        #if tfidf < 0.005: feat_vec['tfidf<0.005'] = 1.0
        
        if tfidf <= 0.05: feat_vec['tfidf<=0.05'] = 1.0
        if tfidf <= 0.02: feat_vec['tfidf<=0.02'] = 1.0
        if tfidf <= 0.01: feat_vec['tfidf<=0.01'] = 1.0
        
#         if tfidf < 0.5: feat_vec['tfidf<0.5'] = 1.0
#         if tfidf < 0.1: feat_vec['tfidf<0.1'] = 1.0
#         if tfidf < 0.05: feat_vec['tfidf<0.05'] = 1.0
#         if tfidf < 0.04: feat_vec['tfidf<0.04'] = 1.0
#         if tfidf < 0.03: feat_vec['tfidf<0.03'] = 1.0
#         if tfidf < 0.02: feat_vec['tfidf<0.02'] = 1.0
#         if tfidf < 0.01: feat_vec['tfidf<0.01'] = 1.0
#         if tfidf < 0.005: feat_vec['tfidf<0.005'] = 1.0
#         if tfidf >= 0.005: feat_vec['tfidf>0.005'] = 1.0
        
        dict[bigram] = feat_vec
        
    return dict

def extract_TFIDF_Rank(prefix, ngram, topK=10):
    BigramTFIDF = fio.LoadDict(prefix + tfidfext, float)
    keys = sorted(BigramTFIDF, key=BigramTFIDF.get, reverse=True)
    
    dict = {}
    for i, bigram in enumerate(keys):
        feat_vec = FeatureVector()
        
        if i < 5: feat_vec['tfidf_rank<5'] = 1.0
        if i < 10: feat_vec['tfidf_rank<10'] = 1.0
#         for k in range(i, topK): 
#             feat_vec['tfidf_rank<=' + str(k)] = 1.0
#         
#         minK = min(topK, i)
#         for k in range(0, minK):
#             feat_vec['tfidf_rank>' + str(k)] = 1.0
            
        dict[bigram] = feat_vec
        
    return dict

def extract_Pos(prefix, ngram):
    BigramPos = fio.LoadDict(prefix + posext, str)
    
    dict = {}
    for bigram, pos in BigramPos.items():
        feat_vec = FeatureVector()
        feat_vec['pos='+pos] = 1.0
        
        if len(bigram.split()) == 2:
            pos2 = pos.split(' ')
            assert(len(pos2) == 2)
            feat_vec['pos_first='+pos2[0]] = 1.0
            feat_vec['pos_second='+pos2[1]] = 1.0
            
        dict[bigram] = feat_vec
        
        
        #TODO
        
    return dict

def extract_inNP(prefix, ngram):
    BigraminNP = fio.LoadDict(prefix + inNPext, str)
    
    dict = {}
    for bigram, inNP in BigraminNP.items():
        
        feat_vec = FeatureVector()
        if inNP == '1': feat_vec['in_NP_Yes'] = 1.0
        if inNP == '0': feat_vec['in_NP_No'] = 1.0
        
        dict[bigram] = feat_vec
        
    return dict

def extract_averageSentenceLength(prefix, ngram):
    IndexPhrase, IndexBigram, PhraseBigram = ILP.getPhraseBigram(prefix + phraseext, Ngram=ngram)
    
    #get word count of phrases
    PhraseBeta = ILP.getWordCounts(IndexPhrase)
    BigramPhrase = ILP.getBigramPhrase(PhraseBigram)
    
    dict = {}
    for bigram, phrases in BigramPhrase.items():
        bigramname = IndexBigram[bigram]
        
        feat_vec = FeatureVector()
        
        ave = 0
        for p in phrases:
            ave += PhraseBeta[p]
        ave /= len(phrases)
        
#         if ave <= 2: feat_vec['ave_sen_length<=2'] = 1.0
#         if ave > 2 and ave <= 4: feat_vec['ave_sen_length=3-4'] = 1.0
#         if ave > 4 and ave <= 6: feat_vec['ave_sen_length=5-6'] = 1.0
#         if ave > 6 and ave <= 8: feat_vec['ave_sen_length=7-8'] = 1.0
        if ave > 5: feat_vec['ave_sen_length=9-10'] = 1.0
        if ave > 10: feat_vec['ave_sen_length=11-15'] = 1.0
        if ave > 15: feat_vec['ave_sen_length=16-20'] = 1.0
        if ave > 20: feat_vec['ave_sen_length=21-25'] = 1.0
        if ave > 25: feat_vec['ave_sen_length=26-30'] = 1.0
        if ave > 30: feat_vec['ave_sen_length>=30'] = 1.0
        
        if ave <= 5: feat_vec['ave_sen_length<=9-10'] = 1.0
        if ave <= 10: feat_vec['ave_sen_length<=11-15'] = 1.0
        if ave <= 15: feat_vec['ave_sen_length<=16-20'] = 1.0
        if ave <= 20: feat_vec['ave_sen_length<=21-25'] = 1.0
        if ave <= 25: feat_vec['ave_sen_length<=26-30'] = 1.0
        if ave <= 30: feat_vec['ave_sen_length<=30'] = 1.0
        
#         if ave > 2: feat_vec['ave_sen_length>2'] = 1.0
#         if ave <= 2 or ave > 4: feat_vec['ave_sen_length<=2,>4'] = 1.0
#         if ave <= 4 or ave > 6: feat_vec['ave_sen_length<=4,>6'] = 1.0
#         if ave <= 6 or ave > 8: feat_vec['ave_sen_length<=6,>8'] = 1.0
#         if ave <= 8 or ave > 10: feat_vec['ave_sen_length<=8,>10'] = 1.0
#         if ave <= 10 or ave > 15: feat_vec['ave_sen_length<=10,>15'] = 1.0
#         if ave <= 15 or ave > 20: feat_vec['ave_sen_length<=15,>20'] = 1.0
#         if ave <= 20 or ave > 25: feat_vec['ave_sen_length<=20,>25'] = 1.0
#         if ave <= 25 or ave > 30: feat_vec['ave_sen_length<=25,>30'] = 1.0
#         if ave <= 30: feat_vec['ave_sen_length<=30'] = 1.0
        
        dict[bigramname] = feat_vec
        
    return dict

def getType(prefix):
    begin_p = prefix.rfind('/')
    name = prefix[begin_p+1:]
    end_p = name.find('.')
    type = name[:end_p]
    return type 

def getLexiconOverlap(s1, s2):
    tokens1 = s1.split()
    tokens2 = s2.split()
    
    unigram1 = ILP.getNgramTokenized(tokens1, 1, NoStopWords=True)
    unigram2 = ILP.getNgramTokenized(tokens2, 1, NoStopWords=True)
    
    overlap = 0
    for w in unigram1:
        if w in unigram2:
            overlap += 1
    return overlap
    
def extract_averageSentenceSimilarity2promp(prefix, ngram):
    type = getType(prefix)
    
    promp = prompts[type] 
    
    IndexPhrase, IndexBigram, PhraseBigram = ILP.getPhraseBigram(prefix + phraseext, Ngram=ngram)
    
    #get word count of phrases
    BigramPhrase = ILP.getBigramPhrase(PhraseBigram)
    
    dict = {}
    for bigram, phrases in BigramPhrase.items():
        bigramname = IndexBigram[bigram]
        
        feat_vec = FeatureVector()
        
        ave = 0
        for p in phrases:
            ave += getLexiconOverlap(IndexPhrase[p], promp)
            
        ave /= len(phrases)
        
        if ave == 0: feat_vec['ave_sen_sim=0'] = 1.0
        if ave <= 1: feat_vec['ave_sen_sim<=1'] = 1.0
        if ave <= 2: feat_vec['ave_sen_sim<=2'] = 1.0
        if ave <= 3: feat_vec['ave_sen_sim<=3'] = 1.0
        if ave <= 4: feat_vec['ave_sen_sim<=4'] = 1.0
        if ave <= 10: feat_vec['ave_sen_sim<=10'] = 1.0
        if ave > 10: feat_vec['ave_sen_sim>10'] = 1.0
        
        if ave != 0: feat_vec['ave_sen_sim!=0'] = 1.0
        if ave > 1: feat_vec['ave_sen_sim>1'] = 1.0
        if ave > 2: feat_vec['ave_sen_sim>2'] = 1.0
        if ave > 3: feat_vec['ave_sen_sim>3'] = 1.0
        if ave > 4: feat_vec['ave_sen_sim>4'] = 1.0
        if ave > 10: feat_vec['ave_sen_sim>10'] = 1.0
        
        dict[bigramname] = feat_vec
        
    return dict
    
def extract_averageSentenceLength_rank(prefix, ngram, topK=10):
    IndexPhrase, IndexBigram, PhraseBigram = ILP.getPhraseBigram(prefix + phraseext, Ngram=ngram)
    
    #get word count of phrases
    PhraseBeta = ILP.getWordCounts(IndexPhrase)
    BigramPhrase = ILP.getBigramPhrase(PhraseBigram)
    
    LengthDict = {}
    
    dict = {}
    for bigram, phrases in BigramPhrase.items():
        bigramname = IndexBigram[bigram]
        
        feat_vec = FeatureVector()
        
        ave = 0
        for p in phrases:
            ave += PhraseBeta[p]
        ave /= len(phrases)
        
        LengthDict[bigramname] = ave
    
    keys = sorted(LengthDict, key=LengthDict.get, reverse=True)
    
    dict = {}
    for i, bigramname in enumerate(keys):
        feat_vec = FeatureVector()
        
        for k in range(i, topK): 
            feat_vec['ave_sen_length_rank<=' + str(k)] = 1.0
        
        minK = min(topK, i)
        for k in range(0, minK):
            feat_vec['ave_sen_length_rank>' + str(k)] = 1.0
        
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
            feat_vec['in_title_Yes'] = 1.0
        else:
            feat_vec['in_title_No'] = 1.0
        
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

def get_word_ratio(bigram):
    words = bigram.split()
    
    stop_wc = 0.0
    for word in words:
        if word in nonwords:
            stop_wc = stop_wc + 1.0
    
    return 1-stop_wc/len(words)

def extract_nonstop_ratio(prefix, ngram):
    phrases, bigrams, PhraseBigram = ILP.getPhraseBigram(prefix + phraseext, Ngram=ngram)
    
    dict = {}
    for bigram in bigrams:
        bigramname = bigrams[bigram]
        
        feat_vec = FeatureVector()
        
        if len(bigramname.split()) == 2:
            words = bigramname.split()
            
            if words[0] in stopwordswithpunctuations:
                feat_vec['stop_first=True'] = 1.0
            if words[1] in stopwordswithpunctuations:
                feat_vec['stop_second=True'] = 1.0
        
#         r = get_nonstop_ratio(bigramname)
#         if r <= 0.5: feat_vec['non_stop_ratio<=0.5'] = 1.0
#         if r > 0.5: feat_vec['non_stop_ratio>0.5'] = 1.0 
        
        dict[bigramname] = feat_vec
        
    return dict

def extract_word_ratio(prefix, ngram):
    phrases, bigrams, PhraseBigram = ILP.getPhraseBigram(prefix + phraseext, Ngram=ngram)
    
    dict = {}
    for bigram in bigrams:
        bigramname = bigrams[bigram]
        
        feat_vec = FeatureVector()
        r = get_word_ratio(bigramname)
        
        if r <= 0.5: feat_vec['word_ratio<=0.5'] = 1.0
        if r > 0.5: feat_vec['word_ratio>0.5'] = 1.0 
        
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
            
            if n1 > 1: feat_vec['freq_firstw>1'] = 1.0 
            if n1 > 2: feat_vec['freq_firstw>2'] = 1.0 
            #if n1 > 3: feat_vec['freq_firstw>3'] = 1.0 
            #if n1 > 4: feat_vec['freq_firstw>4'] = 1.0 
            if n1 > 5: feat_vec['freq_firstw>5'] = 1.0 
            #if n1 > 10: feat_vec['freq_firstw>10'] = 1.0 
            
            if n2 > 1: feat_vec['freq_secondw>1'] = 1.0 
            if n2 > 2: feat_vec['freq_secondw>2'] = 1.0 
            #if n2 > 3: feat_vec['freq_secondw>3'] = 1.0 
            #if n2 > 4: feat_vec['freq_secondw>4'] = 1.0 
            if n2 > 5: feat_vec['freq_secondw>5'] = 1.0 
            #if n2 > 10: feat_vec['freq_secondw>10'] = 1.0 
            
            #if n1 <= 1: feat_vec['freq_firstw<1'] = 1.0 
            if n1 <= 2: feat_vec['freq_firstw<=2'] = 1.0 
            #if n1 < 3: feat_vec['freq_firstw<3'] = 1.0 
            #if n1 < 4: feat_vec['freq_firstw<4'] = 1.0 
            if n1 <= 5: feat_vec['freq_firstw<=5'] = 1.0 
            #if n1 < 10: feat_vec['freq_firstw<10'] = 1.0 
            
            #if n2 < 1: feat_vec['freq_secondw<1'] = 1.0 
            if n2 <= 2: feat_vec['freq_secondw<=2'] = 1.0 
            #if n2 < 3: feat_vec['freq_secondw<3'] = 1.0 
            #if n2 < 4: feat_vec['freq_secondw<4'] = 1.0 
            if n2 <= 5: feat_vec['freq_secondw<=5'] = 1.0 
            #if n2 <= 10: feat_vec['freq_secondw<=10'] = 1.0 
        else:#unigram
            n1 = wf_dict[words[0]]
            
            if n1 > 1: feat_vec['freq_firstw>1'] = 1.0 
            if n1 > 2: feat_vec['freq_firstw>2'] = 1.0 
            #if n1 > 3: feat_vec['freq_firstw>3'] = 1.0 
            #if n1 > 4: feat_vec['freq_firstw>4'] = 1.0 
            if n1 > 5: feat_vec['freq_firstw>5'] = 1.0 
            #if n1 > 10: feat_vec['freq_firstw>10'] = 1.0 
            
            #if n1 < 1: feat_vec['freq_firstw<1'] = 1.0 
            if n1 <= 2: feat_vec['freq_firstw<=2'] = 1.0 
            #if n1 < 3: feat_vec['freq_firstw<3'] = 1.0 
            #if n1 < 4: feat_vec['freq_firstw<4'] = 1.0 
            if n1 <= 5: feat_vec['freq_firstw<=5'] = 1.0 
            #if n1 < 10: feat_vec['freq_firstw<10'] = 1.0
            
        dict[bigramname] = feat_vec
        
    return dict

def extract_ngram_length(prefix, ngram):
    phrases, bigrams, PhraseBigram = ILP.getPhraseBigram(prefix + phraseext, Ngram=ngram)
    
    dict = {}
    for bigram in bigrams:
        bigramname = bigrams[bigram]
        
        feat_vec = FeatureVector()
        if len(bigramname.split()) == 1: feat_vec['ngram=1'] = 1.0 
        if len(bigramname.split()) == 2: feat_vec['ngram=2'] = 1.0 
        
        dict[bigramname] = feat_vec
    return dict

def getPosition(bigram, phrase, bin=5):
    tokens = phrase.lower().split()
    
    if len(bigram.split()) == 2:
        bigrams = ILP.getNgramTokenized(tokens, 2, NoStopWords=False, Stemmed=True)
    else:
        bigrams = ILP.getNgramTokenized(tokens, 1, NoStopWords=False, Stemmed=True)
        
    N = len(bigrams)
    index = bigrams.index(bigram)
    assert(index != -1)

    return util.get_bin_index(index, 0, N, bin)
    
def extract_position(prefix, ngram, bin):
    IndexPhrase, IndexBigram, PhraseBigram = ILP.getPhraseBigram(prefix + phraseext, Ngram=ngram)
    BigramPhrase = ILP.getBigramPhrase(PhraseBigram)
    
    dict = {}
    for bigram, phrases in BigramPhrase.items():
        bigramname = IndexBigram[bigram]
        
        feat_vec = FeatureVector()
        
        for phrase in phrases: #union of the position feature
            p = getPosition(bigramname, IndexPhrase[phrase])
            
            feat_vec['position_'+str(p)] = 1.0
        
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
           
def add_bias(fromdict):
    for k, v in fromdict.items():
        if type(v) == dict or type(v) == FeatureVector:
            fromdict[k]['b'] = 1.0
    return fromdict
           
def extract_single(prefix, ngram, output, titlefile=None, features = None, position_bin = 5):
    data = {}
    
    tf_dict = extract_TF(prefix, ngram)
    tf_rank_dict = extract_TF_Rank(prefix, ngram)
    idftf_dict = extract_TFIDF(prefix, ngram)
    idftf_rank_dict = extract_TFIDF_Rank(prefix, ngram)
    
    pos_dict = extract_Pos(prefix, ngram)
    inNP_dict = extract_inNP(prefix, ngram)
    
    ave_length_dict = extract_averageSentenceLength(prefix, ngram)
    ave_length_dict_rank = extract_averageSentenceLength_rank(prefix, ngram)
    
    ave_sim_dict = extract_averageSentenceSimilarity2promp(prefix, ngram)
    
    one_dict = extract_one(prefix, ngram)  
    stop_ratio_dict = extract_nonstop_ratio(prefix, ngram)
    word_ratio_dict = extract_word_ratio(prefix, ngram)
    
    ngram_length_dict = extract_ngram_length(prefix, ngram)
    frequency_of_words_dict = extract_frequency_of_words(prefix, ngram)
    
    if titlefile != None:
        title_dict = extract_title(prefix, ngram, titlefile)
    else:
        title_dict = {}
    
    position_dict = extract_position(prefix, ngram, bin=position_bin)
    
    if 'tf' in features:    
        data = add_feature_set(data, tf_dict)
    if 'tf_rank' in features:
        data = add_feature_set(data, tf_rank_dict)
    
    if 'idftf' in features:
        data = add_feature_set(data, idftf_dict)
    if 'idftf_rank' in features:
        data = add_feature_set(data, idftf_rank_dict)
    
    if 'pos' in features:
        data = add_feature_set(data, pos_dict)
    
    if 'inNP' in features:
        data = add_feature_set(data, inNP_dict)
            
    if 'ave_length' in features:
        data = add_feature_set(data, ave_length_dict)
    if 'ave_length_rank' in features:
        data = add_feature_set(data, ave_length_dict_rank)
    
    if 'ave_sim' in features:
        data = add_feature_set(data, ave_sim_dict)
    
    if 'stop_word' in features:
        data = add_feature_set(data, stop_ratio_dict)
    if 'word_ratio' in features:
        data = add_feature_set(data, word_ratio_dict)
            
    if 'ngram_length' in features:
        data = add_feature_set(data, ngram_length_dict)
    if 'frequency_of_words' in features:
        data = add_feature_set(data, frequency_of_words_dict)
    
    if 'title' in features:
        data = add_feature_set(data, title_dict)
    
    if 'position' in features:
        data = add_feature_set(data, position_dict)
        
    #data = add_bias(data)
    with open(output, 'w') as outfile:
        json.dump(data, outfile, indent=2)

def extact(ilpdir, np, ngram, features, position_bin):
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
            
            extract_single(prefix, ngram, feature_file, titlefile, features, position_bin)
                    
if __name__ == '__main__':   
    from config import ConfigFile
    config = ConfigFile()
    
    sennadatadir = "../../data/senna/"
    
    for ilpdir in [#"../../data/ILP_Sentence_Supervised_FeatureWeighting/",
                   #"../../data/ILP_Sentence_Supervised_FeatureWeightingAveragePerceptron/",
                   "../../data/ILP_Sentence_Supervised_FeatureWeightingAveragePerceptronMC/",
                   #"../../data/ILP_Sentence_Supervised_FeatureWeightingAveragePerceptronMC/"
                   
                   ]:
        get_ngram_NP.extact_inNP(ilpdir, sennadatadir, np = 'sentence', ngram=config.get_ngrams())
        get_ngram_tfidf.extact_tfidf(ilpdir, np = 'sentence', ngram=config.get_ngrams())
        get_ngram_pos.extact_pos(ilpdir, sennadatadir, np = 'sentence', ngram=config.get_ngrams())
        
        extact(ilpdir, np = 'sentence', ngram=config.get_ngrams(), features=config.get_features(), position_bin = config.get_position_bin())