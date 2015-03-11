import fio
import json
import sys
import porter
import NLTKWrapper
import os
import ILP_baseline as ILP
import NumpyWrapper

ngramTag = "___"

stopwords = [line.lower().strip() for line in fio.ReadFile("../../../Fall2014/summarization/ROUGE-1.5.5/data/smart_common_words.txt")]
punctuations = ['.', '?', '-', ',', '[', ']', '-', ';', '\'', '"', '+', '&', '!', '/', '>', '<', ')', '(', '#', '=']

#Stemming
phraseext = ".key" #a list
studentext = ".keys.source" #json
countext = ".dict"  #a dictionary
lpext = ".lp"
lpsolext = ".sol"

def LoadSVD(svddir, corpusname, K):
    dictname = svddir + "POI_MP_LP_" + str(K) + '_' + corpusname + ".corpus.dict"
    print dictname
    
    uname = svddir + "u_" + str(K) + '_' + corpusname + ".mat.txt"
    
    dict = fio.LoadDict(dictname, 'int')
    mat = fio.ReadFile(uname)
    assert(len(dict) == len(mat))
    
    U = {}
    
    for k, v in dict.items():
        k = k.replace(ngramTag, " ")
        
        #stemming
        k = porter.getStemming(k)
        
        U[k] = [float(x) for x in mat[v].split()]
    
    return U

def ProcessCountDict(CountDict):
    newCountDict = {}
    
    for phrase, count in CountDict.items():
        phrase = ILP.ProcessLine(phrase)
        newCountDict[phrase] = count
    
    return newCountDict
        
def getBigramWeight_SVD(PhraseBigram, IndexBigram, PhraseIndex, CountFile, svddir, corpusname, K):
    #load the svd
    U = LoadSVD(svddir, corpusname, K)
    
    BigramCount = {}
    CountDict = fio.LoadDict(CountFile, 'float')
    
    #CountDict = ProcessCountDict(CountDict)
    
    for phrase, bigrams in PhraseBigram.items():
        assert(phrase in PhraseIndex)
        p = PhraseIndex[phrase]
        try:
            fequency = CountDict[p]
        except Exception as e:
            print p
            exit()

        for bigram in bigrams:
            if bigram not in BigramCount:
                BigramCount[bigram] = 0
            BigramCount[bigram] = BigramCount[bigram] + fequency
    
    #BigramTheta = BigramCount
    
    BigramTheta = {}
             
    for bigram1 in BigramCount.keys():
        bigram1name = IndexBigram[bigram1]
        weight = BigramCount[bigram1]
         
        if bigram1name in U:    
            for bigram2 in BigramCount.keys():
                if bigram2 == bigram1: continue
                bigram2name = IndexBigram[bigram2]
                if bigram2name not in U: continue
                 
                sim = NumpyWrapper.cosine_similarity(U[bigram1name], U[bigram2name])
                #if sim < 0: continue #no penalty
                
                weight = weight + sim
         
        BigramTheta[bigram1] = weight
        
    return BigramTheta
            
def ILP1(prefix, L, svddir, corpusname, K):
    # get each stemmed bigram, sequence the bigram and the phrase
    # bigrams: {index:bigram}, a dictionary of bigram index, X
    # phrases: {index:phrase}, is a dictionary of phrase index, Y
    #PhraseBigram: {phrase, [bigram]}
    IndexPhrase, IndexBigram, PhraseBigram = ILP.getPhraseBigram(prefix + phraseext)
    fio.SaveDict(IndexPhrase, prefix + ".phrase_index.dict")
    fio.SaveDict(IndexBigram, prefix + ".bigram_index.dict")
    
    #get weight of bigrams
    BigramTheta = getBigramWeight_SVD(PhraseBigram, IndexBigram, IndexPhrase, prefix + countext, svddir, corpusname, K) # return a dictionary
    WeightDict = {}
    
    for bigram, theta in BigramTheta.items():
        WeightDict[IndexBigram[bigram]] = theta
    fio.SaveDict(WeightDict, prefix + ".weight.dict")
    
    #get word count of phrases
    PhraseBeta = ILP.getWordCounts(IndexPhrase)
       
    #get {bigram:[phrase]} dictionary
    BigramPhrase = ILP.getBigramPhrase(PhraseBigram)
    
    lpfile = prefix
    ILP.formulateProblem(BigramTheta, PhraseBeta, BigramPhrase, PhraseBigram, L, lpfile)
    
    m = ILP.SloveILP(lpfile)
    
    output = lpfile + '.L' + str(L) + ".summary"
    ILP.ExtractSummaryfromILP(lpfile, IndexPhrase, output)
    
def ILP_Summarizer(ilpdir, np, L, svddir, corpusname, K):
    sheets = range(0,12)
    
    for i, sheet in enumerate(sheets):
        week = i + 1
        dir = ilpdir + str(week) + '/'
        
        for type in ['POI', 'MP', 'LP']:
            prefix = dir + type + "." + np
            
            ILP1(prefix, L, svddir, corpusname, K)
            
if __name__ == '__main__':
    ilpdir = "../../data/ILP_Sentence_SVD/"
    svddir = "../../data/SVD_Sentence/"
    corpusname = "corpus"
    K = 50    
    #ILP1(ilpdir + "test/MP.syntax", 10)
    
    for L in [30]:
        for np in ['sentence']:
            ILP_Summarizer(ilpdir, np, L, svddir, corpusname, K)

    print "done"