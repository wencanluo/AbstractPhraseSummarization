import fio
import json
import sys
import porter
import NLTKWrapper
import os
import numpy
import NumpyWrapper

import ILP_baseline
import ILP_SVD
import ILP_MC
import ILP_Supervised_FeatureWeight
import ILP_Supervised_MC

from ILP_Supervised_FeatureWeight import minthreshold
from ILP_Supervised_FeatureWeight import maxIter

from feat_vec import FeatureVector

#Stemming
phraseext = ".key" #a list
studentext = ".keys.source" #json
countext = ".dict"  #a dictionary
lpext = ".lp"
lpsolext = ".sol"
sumexe = ".ref.summary"
featureext = ".f"

ngramTag = "___"
    
def formulateProblem(IndexBigram, Lambda, Weights, PhraseBeta, PhraseBigram, BigramPhrase, partialBigramPhrase, partialPhraseBigram, L, lpfileprefix, FeatureVecU):
    SavedStdOut = sys.stdout
    sys.stdout = open(lpfileprefix + lpext, 'w')

    #write objective
    print "Maximize"
    objective = []
    for bigram in BigramPhrase.keys():
        bigramname = IndexBigram[bigram]
        
        if bigramname in FeatureVecU:
            fvec = FeatureVector(FeatureVecU[bigramname])
            w = Weights.dot(fvec) - minthreshold
            if w <= 0: continue
            objective.append(" ".join([str(w*Lambda), bigram]))
        else:
            print "bigramname not in FeatureVecU", bigramname
            
    print "  ", " + ".join(objective)
    
    #write constraints
    print "Subject To"
    ILP_MC.WriteConstraint1(PhraseBeta, L)
    
    ILP_MC.WriteConstraint2(partialBigramPhrase)
    
    ILP_MC.WriteConstraint3(partialPhraseBigram)
       
    indicators = []
    for bigram in BigramPhrase.keys():
        indicators.append(bigram)
    for phrase in PhraseBeta.keys():
        indicators.append(phrase)
        
    #write Bounds
    print "Bounds"
    for indicator in indicators:
        print "  ", indicator, "<=", 1
    
    indicators = []
    #for bigram in partialBigramPhrase.keys():
    #    indicators.append(bigram)
    for phrase in PhraseBeta.keys():
        indicators.append(phrase)
        
    #write Integers
    print "Integers"
    print "  ", " ".join(indicators)
    
    #write End
    print "End"
    sys.stdout = SavedStdOut
        
def ILP_Supervised(Weights, prefix, featurefile, svdfile, svdpharefile, L, Lambda, ngram, MalformedFlilter, prefixA, threshold):
    # get each stemmed bigram, sequence the bigram and the phrase
    # bigrams: {index:bigram}, a dictionary of bigram index, X
    # phrases: {index:phrase}, is a dictionary of phrase index, Y
    #PhraseBigram: {phrase, [bigram]}
    IndexPhrase, IndexBigram, PhraseBigram = ILP_baseline.getPhraseBigram(prefix+phraseext, Ngram=ngram, MalformedFlilter=MalformedFlilter)
    fio.SaveDict(IndexPhrase, prefix + ".phrase_index.dict")
    fio.SaveDict(IndexBigram, prefix + ".bigram_index.dict")
    
    #get weight of bigrams {bigram:weigth}
    #BigramTheta = Weights #ILP.getBigramWeight_TF(PhraseBigram, phrases, prefix + countext) # return a dictionary
    
    #get word count of phrases
    PhraseBeta = ILP_baseline.getWordCounts(IndexPhrase)
    
    #get {bigram:[phrase]} dictionary
    BigramPhrase = ILP_baseline.getBigramPhrase(PhraseBigram)
    
    partialPhraseBigram, PartialBigramPhrase = ILP_MC.getPartialPhraseBigram(IndexPhrase, IndexBigram, prefix + phraseext, svdfile, svdpharefile, threshold=threshold)
    fio.SaveDict2Json(partialPhraseBigram, prefix + ".partialPhraseBigram.dict")
    fio.SaveDict2Json(PartialBigramPhrase, prefix + ".PartialBigramPhrase.dict")
        
    FeatureVecU = ILP_Supervised_FeatureWeight.LoadFeatureSet(featurefile)
    
    lpfile = prefix
    formulateProblem(IndexBigram, Lambda, Weights, PhraseBeta, PhraseBigram, BigramPhrase, PartialBigramPhrase, partialPhraseBigram, L, lpfile, FeatureVecU)
    
    m = ILP_baseline.SloveILP(lpfile)
    
    output = lpfile + '.L' + str(L) + "." + str(Lambda) + ".summary"
    ILP_baseline.ExtractSummaryfromILP(lpfile, IndexPhrase, output)

def TestILP(train, test, ilpdir, svddir, np, L, Lambda, ngram, MalformedFlilter, featuredir, prefixA, threshold):
    Weights = {}
    BigramIndex = {}
    
    round = 0
    for round in range(maxIter):
        weightfile = ilpdir + str(round) + '_' + '_'.join(train) + '_weight_' + "_" + '.json'
        if not fio.IsExist(weightfile):
            break
    round = round - 1
    
    weightfile = ilpdir + str(round) + '_' + '_'.join(train) + '_weight_' + "_" + '.json'
    #bigramfile = ilpdir + str(round) + '_' + '_'.join(train) + '_bigram_' + "_" + '.json'
    
    print weightfile
    
    with open(weightfile, 'r') as fin:
        Weights = FeatureVector(json.load(fin, encoding="utf-8"))   
            
    #BigramIndex = fio.LoadDict(bigramfile, "str")
        
    for sheet in test:
        week = int(sheet) + 1
        dir = ilpdir + str(week) + '/'
        
        for type in ['POI', 'MP', 'LP']:
            prefix = dir + type + "." + np
            print "Test: ", prefix
            
            svdfile = svddir + str(week) + '/' + type + prefixA
            svdpharefile = svddir + str(week) + '/' + type + '.' + np + ".key"
            
            featurefile = featuredir + str(week) + '/' + type + featureext
            
            ILP_Supervised(Weights, prefix, featurefile, svdfile, svdpharefile, L, Lambda, ngram, MalformedFlilter, prefixA, threshold)

def ILP_CrossValidation(ilpdir, svddir, np, L, Lambda, ngram, MalformedFlilter, featuredir, prefixA=".org.softA", threshold=1.0):
    for train, test in LeaveOneLectureOutPermutation():
        ILP_Supervised_FeatureWeight.TrainILP(train, ilpdir, np, L, Lambda, ngram, MalformedFlilter, featuredir)
        TestILP(train, test, ilpdir, svddir, np, L, Lambda, ngram, MalformedFlilter, featuredir, prefixA, threshold)

def LeaveOneLectureOutPermutation():
    sheets = range(0,12)
    N = len(sheets)
    for i in range(N):
        train = [str(k) for k in range(N) if k != i]
        #train = [str(i)]
        test = [str(i)]
        yield train, test
            
if __name__ == '__main__':   
    #ilpdir = "../../data/ILP_Sentence_Supervised_Oracle/"
    #ilpdir = "../../data/ILP_Sentence_Supervised_SVD_BOOK/"
    ilpdir = "../../data/ILP_Sentence_Supervised_FeatureWeightingMC/"
    
    featuredir = ilpdir
    
    svddir = "../../data/SVD_Sentence/"
    #svddir = ilpdir
    
    corpusname = "book"
    K = 50
       
    MalformedFlilter = False
    ngrams = [1,2]
    
    #ILP_baseline.SloveILP(ilpdir + "3/MP.sentence")
    
    #for Lambda in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
    for Lambda in [1.0]:
         #for L in [10, 15, 20, 25, 30, 35, 40, 45, 50]:
         for L in [30]:
             for np in ['sentence']: #'chunk
                 for iter in range(1):
                    ILP_CrossValidation(ilpdir, svddir, np, L, Lambda, ngrams, MalformedFlilter, featuredir, prefixA=".200_2.softA", threshold=0.5)
    
    print "done"