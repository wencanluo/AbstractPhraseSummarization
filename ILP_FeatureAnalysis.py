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

import ILP_Supervised_FeatureWeight
import ILP_baseline as ILP

from feat_vec import FeatureVector 

from ILP_baseline import stopwords
import matplotlib.pyplot as plt

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
        dict[bigramname] = tf
        
    return dict

def extract_TF_Rank(prefix, ngram, topK=10):
    phrases, bigrams, PhraseBigram = ILP.getPhraseBigram(prefix + phraseext, Ngram=ngram)
    
    #get weight of bigrams {bigram:weigth}
    BigramTheta = ILP.getBigramWeight_TF(PhraseBigram, phrases, prefix + countext) # return a dictionary
    
    keys = sorted(BigramTheta, key=BigramTheta.get, reverse=True)
    
    dict = {}
    for i, bigram in enumerate(keys):
        bigramname = bigrams[bigram]
        
        dict[bigramname] = i
        
    return dict

def extract_TFIDF(prefix, ngram):
    BigramTFIDF = fio.LoadDict(prefix + tfidfext, float)
    
    dict = {}
    for bigram, tfidf in BigramTFIDF.items():
        
        dict[bigram] = tfidf
        
    return dict

def extract_TFIDF_Rank(prefix, ngram, topK=10):
    BigramTFIDF = fio.LoadDict(prefix + tfidfext, float)
    keys = sorted(BigramTFIDF, key=BigramTFIDF.get, reverse=True)
    
    dict = {}
    for i, bigram in enumerate(keys):
        feat_vec = FeatureVector()
        
        for k in range(i, topK): 
            feat_vec['tfidf_rank<=' + str(k)] = 1.0
        
        dict[bigram] = feat_vec
        
    return dict

def extract_Pos(prefix, ngram):
    BigramPos = fio.LoadDict(prefix + posext, str)
    
    dict = {}
    for bigram, pos in BigramPos.items():
        dict[bigram] = pos
        
    return dict

def extract_inNP(prefix, ngram):
    BigraminNP = fio.LoadDict(prefix + inNPext, str)
    
    dict = {}
    for bigram, inNP in BigraminNP.items():
        
        dict[bigram] = int(inNP)
        
    return dict

def extract_averageSentenceLength(prefix, ngram):
    IndexPhrase, IndexBigram, PhraseBigram = ILP.getPhraseBigram(prefix + phraseext, Ngram=[1,2])
    
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
        
        dict[bigramname] = ave
        
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
    
    IndexPhrase, IndexBigram, PhraseBigram = ILP.getPhraseBigram(prefix + phraseext, Ngram=[1,2])
    
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
        
        dict[bigramname] = ave
        
    return dict
    
def extract_averageSentenceLength_rank(prefix, ngram, topK=10):
    IndexPhrase, IndexBigram, PhraseBigram = ILP.getPhraseBigram(prefix + phraseext, Ngram=[1,2])
    
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
            feat_vec['ave_sen_length_rank=' + str(k)] = 1.0
        
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
            dict[bigramname] = 1.0
        else:
            dict[bigramname] = 0.0
        
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
                
        dict[bigramname] = r
        
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
            
            if n1 == 0:feat_vec['freq_firstw=0'] = 1.0 
            if n1 >= 1: feat_vec['freq_firstw>=1'] = 1.0 
            if n1 >= 2: feat_vec['freq_firstw>=2'] = 1.0 
            if n1 >= 3: feat_vec['freq_firstw>=3'] = 1.0 
            if n1 >= 4: feat_vec['freq_firstw>=4'] = 1.0 
            if n1 >= 5: feat_vec['freq_firstw>=5'] = 1.0 
            if n1 >= 10: feat_vec['freq_firstw>=10'] = 1.0 
            
            if n2 == 0: feat_vec['freq_secondw=0'] = 1.0 
            if n2 >= 1: feat_vec['freq_secondw>=1'] = 1.0 
            if n2 >= 2: feat_vec['freq_secondw>=2'] = 1.0 
            if n2 >= 3: feat_vec['freq_secondw>=3'] = 1.0 
            if n2 >= 4: feat_vec['freq_secondw>=4'] = 1.0 
            if n2 >= 5: feat_vec['freq_secondw>=5'] = 1.0 
            if n2 >= 10: feat_vec['freq_secondw>=10'] = 1.0 
        else:#unigram
            n1 = wf_dict[words[0]]
            
            if n1 == 0: feat_vec['freq_firstw=0'] = 1.0 
            if n1 >= 1: feat_vec['freq_firstw>=1'] = 1.0 
            if n1 >= 2: feat_vec['freq_firstw>=2'] = 1.0 
            if n1 >= 3: feat_vec['freq_firstw>=3'] = 1.0 
            if n1 >= 4: feat_vec['freq_firstw>=4'] = 1.0 
            if n1 >= 5: feat_vec['freq_firstw>=5'] = 1.0 
            if n1 >= 10: feat_vec['freq_firstw>=10'] = 1.0 
            
        dict[bigramname] = feat_vec
        
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
            todict[k] = {}
        if type(v) == dict or type(v) == FeatureVector:
            for name, val in v.iteritems():
                todict[k][name] = val
    return todict

               
def extract_single(prefix, ngram, output, titlefile=None):
    
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
    ngram_length_dict = extract_ngram_length(prefix, ngram)
    frequency_of_words_dict = extract_frequency_of_words(prefix, ngram)

    if titlefile != None:
        title_dict = extract_title(prefix, ngram, titlefile)
    else:
        title_dict = {}

    return ave_sim_dict    

def plotDict(dict):
    keys = dict.keys()
    
    max = int(numpy.max(keys))
    min = numpy.min(keys)
    min = int(min) if min > 0 else 0 
    
    x =  range(min, max+1)
    
    y = []
    
    for v in x:
        if v in dict:
            y.append(dict[v])
        else:
            y.append(0)
    
    plt.plot(x, y)
    plt.show()

def plothist(values):
    plt.hist(values)
    plt.show()
    
def feature_analyze(ilpdir, np, ngram):
    sheets = range(0,12)
    
    #count = defaultdict(int)
    values = []
    values_pos = []
    values_neg = []

    
    for i, sheet in enumerate(sheets):
        week = i + 1
        dir = ilpdir + str(week) + '/'
        
        for type in ['POI', 'MP', 'LP']:
            prefix = dir + type
            feature_file = prefix + featureext
            
            prefix = prefix + '.' + np
            print feature_file 
            
            titlefile = titledir + str(week) + '.TXT'
            reffile = ILP_Supervised_FeatureWeight.ExtractRefSummaryPrefix(prefix) + '.ref.summary'
            _, IndexRefBigram, SummaryRefBigram = ILP.getPhraseBigram(reffile, Ngram=ngram, MalformedFlilter=False)
            RefBigramDict = ILP_Supervised_FeatureWeight.getBigramDict(IndexRefBigram, SummaryRefBigram)
            
            data = extract_single(prefix, ngram, feature_file, titlefile)
            
            for bigram, v in data.items():
                #count[v] = count[v] + 1
                if bigram in RefBigramDict:
                    values_pos.append(v)
                else:
                    values_neg.append(v)
                    
                values.append(v)
    
    #values = [v for v in values if v != 1]
    plothist(values)
    plothist(values_pos)
    plothist(values_neg)
    
if __name__ == '__main__':   
    ilpdir = "../../data/ILP_Sentence_Supervised_FeatureWeighting/"
    
    feature_analyze(ilpdir, np = 'sentence', ngram=[1,2])
    