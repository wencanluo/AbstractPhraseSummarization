import fio
import json
import sys
import porter
import NLTKWrapper
import os
import numpy

import ILP_baseline as ILP


#Stemming
phraseext = ".key" #a list
studentext = ".keys.source" #json
countext = ".dict"  #a dictionary
lpext = ".lp"
lpsolext = ".sol"
sumexe = ".ref.summary"
    
def formulateProblem(Lambda, StudentGamma, StudentPhrase, BigramTheta, PhraseBeta, BigramPhrase, PhraseBigram, L, lpfileprefix):
    SavedStdOut = sys.stdout
    sys.stdout = open(lpfileprefix + lpext, 'w')
    
    if Lambda== None:
        Lambda = 1.0
        
    #write objective
    print "Maximize"
    objective = []
    for bigram, theta in BigramTheta.items():
        if theta == 0: continue
        
        objective.append(" ".join([str(theta*Lambda), bigram]))
    for student, grama in StudentGamma.items():
        objective.append(" ".join([str(grama*(1-Lambda)), student]))
    print "  ", " + ".join(objective)
    
    #write constraints
    print "Subject To"
    ILP.WriteConstraint1(PhraseBeta, L)
    
    ILP.WriteConstraint2(BigramPhrase)
    
    ILP.WriteConstraint3(PhraseBigram)
    
    ILP.WriteConstraint4(StudentPhrase)
    
    indicators = []
    for bigram in BigramTheta.keys():
        indicators.append(bigram)
    for phrase in PhraseBeta.keys():
        indicators.append(phrase)
    for student in StudentGamma.keys():
        indicators.append(student)
        
    #write Bounds
    print "Bounds"
    for indicator in indicators:
        print "  ", indicator, "<=", 1
    
    #write Integers
    print "Integers"
    print "  ", " ".join(indicators)
    
    #write End
    print "End"
    sys.stdout = SavedStdOut

def UpdatePhraseBigram(BigramIndex, phrasefile, Ngram=[2], MalformedFlilter=False):
    #get phrases
    lines = fio.ReadFile(phrasefile)
    phrases = [line.strip() for line in lines]
    
    newPhrases = []
    for phrase in phrases:
        if MalformedFlilter:
            if ILP.isMalformed(phrase.lower()): 
                print phrase
            else:
                newPhrases.append(phrase)
    
    if MalformedFlilter:
        phrases = newPhrases
    
    PhraseBigram = {}
    
    #get index of phrase
    j = 1
    phraseIndex = {}
    for phrase in phrases:
        if phrase not in phraseIndex:
            index = 'Y' + str(j)
            phraseIndex[phrase] = index
            PhraseBigram[index] = []
            j = j + 1
    
    #get bigram index and PhraseBigram
    i = 1
    for phrase in phrases:
        pKey = phraseIndex[phrase]
        
        phrase = porter.getStemming(phrase)
        tokens = phrase.lower().split()

        ngrams = []
        for n in Ngram:
            grams = ILP.getNgramTokenized(tokens, n, NoStopWords=True)
            ngrams = ngrams + grams
            
        for bigram in ngrams:
            if bigram not in BigramIndex: continue
            bKey = BigramIndex[bigram]
            
            PhraseBigram[pKey].append(bKey)
    
    IndexPhrase = {}
    for k,v in phraseIndex.items():
        IndexPhrase[v] = k
    
    IndexBigram = {}
    for k,v in BigramIndex.items():
        IndexBigram[v] = k
        
    return IndexPhrase, IndexBigram, PhraseBigram
        
def ILP_Supervised(BigramIndex, Weights, prefix, L, Lambda, ngram, MalformedFlilter):
    # get each stemmed bigram, sequence the bigram and the phrase
    # bigrams: {index:bigram}, a dictionary of bigram index, X
    # phrases: {index:phrase}, is a dictionary of phrase index, Y
    #PhraseBigram: {phrase, [bigram]}
    phrases, bigrams, PhraseBigram = UpdatePhraseBigram(BigramIndex, prefix + phraseext, Ngram=ngram, MalformedFlilter=MalformedFlilter)
    fio.SaveDict(phrases, prefix + ".phrase_index.dict")
    fio.SaveDict(bigrams, prefix + ".bigram_index.dict")
    
    #get weight of bigrams {bigram:weigth}
    BigramTheta = Weights #ILP.getBigramWeight_TF(PhraseBigram, phrases, prefix + countext) # return a dictionary
    
    #get word count of phrases
    PhraseBeta = ILP.getWordCounts(phrases)
    
    #get {bigram:[phrase]} dictionary
    BigramPhrase = ILP.getBigramPhrase(PhraseBigram)

    #get {student:phrase}
    #sequence students, students = {index:student}
    students, StudentPhrase = ILP.getStudentPhrase(phrases, prefix + studentext)
    fio.SaveDict(students, prefix + ".student_index.dict")
    
    #get {student:weight0}
    StudentGamma = ILP.getStudentWeight_One(StudentPhrase)
    
    lpfile = prefix
    formulateProblem(Lambda, StudentGamma, StudentPhrase, BigramTheta, PhraseBeta, BigramPhrase, PhraseBigram, L, lpfile)
    
    m = ILP.SloveILP(lpfile)
    
    if Lambda == None:
        output = lpfile + '.L' + str(L) + ".summary"
    else:
        output = lpfile + '.L' + str(L) + "." + str(Lambda) + ".summary"
    ILP.ExtractSummaryfromILP(lpfile, phrases, output)

def getLastIndex(BigramIndex):
    maxI = 1
    for bigram in BigramIndex.values():
        if int(bigram[1:]) > maxI:
            maxI = int(bigram[1:])
    return maxI

def UpdateWeight(BigramIndex, Weights, sumprefix, prefix, L, Lambda, ngram, MalformedFlilter):
    # the weights of the bigram is the frequency appear in the golden summary
    #read the summary
    _, IndexBigram, SummaryBigram = ILP.getPhraseBigram(sumprefix + sumexe, Ngram=ngram, MalformedFlilter=MalformedFlilter)
    _, IndexBigramResponse, PhraseBigram = ILP.getPhraseBigram(prefix + phraseext, Ngram=ngram, MalformedFlilter=MalformedFlilter)    
    
    i = getLastIndex(BigramIndex)
    
    #get the bigrams
    for summary, bigrams in SummaryBigram.items():
        for bigram in bigrams:
            bigramname = IndexBigram[bigram]
            if bigramname not in BigramIndex:
                bindex = 'X' + str(i)
                i = i + 1
                BigramIndex[bigramname] = bindex
            else:
                bindex = BigramIndex[bigramname]
                 
            #update the weights
            if bindex not in Weights:
                Weights[bindex] = 0
            
            Weights[bindex] = Weights[bindex] + 1
    
#     for phrase, bigrams in PhraseBigram.items():
#         for bigram in bigrams:
#             bigramname = IndexBigramResponse[bigram]
#             if bigramname not in BigramIndex: continue
#                 
#             bindex = BigramIndex[bigramname]
#                   
#             #update the weights
#             if bindex not in Weights:
#                 Weights[bindex] = 0
#              
#             Weights[bindex] = Weights[bindex] + 1

def TrainILP(train, ilpdir, np, L, Lambda, ngram, MalformedFlilter):
    Weights = {} #{Index:Weight}
    BigramIndex = {} #{bigram:index}
    
    weightfile = ilpdir + '_'.join(train) + '_weight.json'
    bigramfile = ilpdir + '_'.join(train) + '_bigram.json'
                
    for sheet in train:
        week = int(sheet) + 1
        dir = ilpdir + str(week) + '/'
        
        for type in ['POI', 'MP', 'LP']:
            prefix = dir + type + "." + np
            summprefix = dir + type
            
            UpdateWeight(BigramIndex, Weights, summprefix, prefix, L, Lambda, ngram, MalformedFlilter)
            #ILP3(prefix, L, Lambda, ngram, MalformedFlilter)
        
    fio.SaveDict(Weights, weightfile, True)
    fio.SaveDict(BigramIndex, bigramfile)

def TestILP(train, test, ilpdir, np, L, Lambda, ngram, MalformedFlilter):
    Weights = {}
    BigramIndex = {}
    
    weightfile = ilpdir + '_'.join(train) + '_weight.json'
    Weights = fio.LoadDict(weightfile, "float")
            
    bigramfile = ilpdir + '_'.join(train) + '_bigram.json'
    BigramIndex = fio.LoadDict(bigramfile, "str")
    
    for sheet in test:
        week = int(sheet) + 1
        dir = ilpdir + str(week) + '/'
        
        for type in ['POI', 'MP', 'LP']:
            prefix = dir + type + "." + np
            ILP_Supervised(BigramIndex, Weights, prefix, L, Lambda, ngram, MalformedFlilter)

def ILP_CrossValidation(ilpdir, np, L, Lambda, ngram, MalformedFlilter):
    for train, test in LeaveOneLectureOutPermutation():
        TrainILP(train, ilpdir, np, L, Lambda, ngram, MalformedFlilter)
        TestILP(train, test, ilpdir, np, L, Lambda, ngram, MalformedFlilter)

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
    ilpdir = "../../data/ILP_Sentence_Supervised/"
    
    #for Lambda in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
    for Lambda in [None]:
         #for L in [10, 15, 20, 25, 30, 35, 40, 45, 50]:
         for L in [30]:
             for np in ['sentence', ]: #'chunk
                 ILP_CrossValidation(ilpdir, np, L, Lambda, ngram=[1,2], MalformedFlilter=False)

    print "done"