import fio
import json
import sys
import porter
import NLTKWrapper
import os
import numpy
import NumpyWrapper

import ILP_baseline as ILP
import ILP_SVD

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

global FeatureVecU
FeatureVecU = {}

def LoadFeatureSet(featurename):
    with open(featurename, 'r') as fin:
        featureV = json.load(fin)
        
    return featureV
    
def formulateProblem(bigrams, Lambda, StudentGamma, StudentPhrase, Weights, PhraseBeta, BigramPhrase, PhraseBigram, L, lpfileprefix):
    SavedStdOut = sys.stdout
    sys.stdout = open(lpfileprefix + lpext, 'w')

    #write objective
    print "Maximize"
    objective = []
    for bigram in BigramPhrase:
        bigramname = bigrams[bigram]
        
        if bigramname in FeatureVecU:
            fvec = FeatureVector(FeatureVecU[bigramname])
            w = Weights.dot(fvec)
            objective.append(" ".join([str(w*Lambda), bigram]))
            
    for student, grama in StudentGamma.items():
        if Lambda==1:continue
        
        objective.append(" ".join([str(grama*(1-Lambda)), student]))
    
    print "  ", " + ".join(objective)
    
    #write constraints
    print "Subject To"
    ILP.WriteConstraint1(PhraseBeta, L)
    
    ILP.WriteConstraint2(BigramPhrase)
    
    ILP.WriteConstraint3(PhraseBigram)
    
    ILP.WriteConstraint4(StudentPhrase)
    
    indicators = []
    for bigram in BigramPhrase.keys():
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
        
        #get stemming
        phrase = porter.getStemming(phrase)
        
        #get bigrams
        ngrams = []
        for n in Ngram:
            grams = NLTKWrapper.getNgram(phrase, n)
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
        
def ILP_Supervised(Weights, prefix, L, Lambda, ngram, MalformedFlilter):
    # get each stemmed bigram, sequence the bigram and the phrase
    # bigrams: {index:bigram}, a dictionary of bigram index, X
    # phrases: {index:phrase}, is a dictionary of phrase index, Y
    #PhraseBigram: {phrase, [bigram]}
    phrases, bigrams, PhraseBigram = ILP.getPhraseBigram(prefix+phraseext, Ngram=ngram, MalformedFlilter=MalformedFlilter)
    fio.SaveDict(phrases, prefix + ".phrase_index.dict")
    fio.SaveDict(bigrams, prefix + ".bigram_index.dict")
    
    #get weight of bigrams {bigram:weigth}
    #BigramTheta = Weights #ILP.getBigramWeight_TF(PhraseBigram, phrases, prefix + countext) # return a dictionary
    
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
    formulateProblem(bigrams, Lambda, StudentGamma, StudentPhrase, Weights, PhraseBeta, BigramPhrase, PhraseBigram, L, lpfile)
    
    m = ILP.SloveILP(lpfile)
    
    output = lpfile + '.L' + str(L) + "." + str(Lambda) + ".summary"
    ILP.ExtractSummaryfromILP(lpfile, phrases, output)

def getLastIndex(BigramIndex):
    maxI = 1
    for bigram in BigramIndex.values():
        if int(bigram[1:]) > maxI:
            maxI = int(bigram[1:])
    return maxI

def InitializeWeight():
    # the weights of the feature functions are 0
    Weights = FeatureVector()
    return Weights

def ExtractRefSummaryPrefix(prefix):
    key = prefix.rfind('.')
    if key==-1:
        return prefix
    return prefix[:key]

def getBigramDict(IndexBigram, PhraseBigram):
    dict = {}
    for phrase, bigrams in PhraseBigram.items():
        for bigram in bigrams:
            bigramname = IndexBigram[bigram]
            
            if bigramname not in dict:
                dict[bigramname] = 0
            dict[bigramname] = dict[bigramname] + 1
    return dict

def generate_randomsummary(prefix, L, sumfile):
    print "no summary is found, generating random ones"
    lines = fio.ReadFile(prefix + phraseext)
    lines = [line.strip() for line in lines]
    
    index = numpy.random.permutation(len(lines))
    
    summaries = []
    
    length = 0
    for i in index:
        line = lines[i]
        length += len(line.split())
        
        if length <= L:
            summaries.append(line)
        else:
            break
    
    fio.SaveList(summaries, sumfile)

def UpdateWeight(BigramIndex, Weights, prefix, L, Lambda, ngram, MalformedFlilter, featurefile):
    ILP_Supervised(Weights, prefix, L, Lambda, ngram, MalformedFlilter)
    
    #read the summary, update the weight 
    sumfile = prefix + '.L' + str(L) + "." + str(Lambda) + '.summary'
    
    if len(fio.ReadFile(sumfile)) == 0:#no summary is generated, using a random baseline
        generate_randomsummary(prefix, L, sumfile)
    
    _, IndexBigram, SummaryBigram = ILP.getPhraseBigram(sumfile, Ngram=ngram, MalformedFlilter=MalformedFlilter)
    
    reffile = ExtractRefSummaryPrefix(prefix) + '.ref.summary'
    _, IndexRefBigram, SummaryRefBigram = ILP.getPhraseBigram(reffile, Ngram=ngram, MalformedFlilter=MalformedFlilter)
    
    RefBigramDict = getBigramDict(IndexRefBigram, SummaryRefBigram)
    
    #update the weights
    global FeatureVecU
    if FeatureVecU == {}:
        FeatureVecU = LoadFeatureSet(featurefile)
    
    i = getLastIndex(BigramIndex)
    
    #if the generated summary matches the golden summary, update the bigrams
    
    #get the bigrams
    #{sentence:bigrams}
    for summary, bigrams in SummaryBigram.items():
        for bigram in bigrams:
            bigramname = IndexBigram[bigram]
            if bigramname not in FeatureVecU: continue
            
            vec = FeatureVector(FeatureVecU[bigramname])
            
            if bigramname in RefBigramDict:
                Weights += vec
            else:
                Weights = Weights.sub_cutoff(vec*0.1)
                
    return Weights

def TrainILP(train, ilpdir, np, L, Lambda, ngram, MalformedFlilter, svddir):
    Weights = {} #{Index:Weight}
    BigramIndex = {} #{bigram:index}
    
    round = 0
    for round in range(10):
        weightfile = ilpdir + str(round) + '_' + '_'.join(train) + '_weight_' + "_" + '.json'
        if not fio.IsExist(weightfile):
            break
    
    if round != 0:
        nextround = round
        round = round -1
        weightfile = ilpdir + str(round) + '_' + '_'.join(train) + '_weight_' + "_" + '.json'
        #bigramfile = ilpdir + str(round) + '_' + '_'.join(train) + '_bigram_' + "_" + '.json'
    
        with open(weightfile, 'r') as fin:
            Weights = FeatureVector(json.load(fin, encoding="utf-8"))
                
        #BigramIndex = fio.LoadDict(bigramfile, "str")
    else:
        nextround = 0
    
    firstRound = False
    
    for round in range(nextround, nextround+1):
        weightfile = ilpdir + str(round) + '_' + '_'.join(train) + '_weight_'  + "_" + '.json'
        bigramfile = ilpdir + str(round) + '_' + '_'.join(train) + '_bigram_'  + "_" + '.json'
    
        for sheet in train:
            week = int(sheet) + 1
            dir = ilpdir + str(week) + '/'
            
            for type in ['POI', 'MP', 'LP']:
                prefix = dir + type + "." + np
                summprefix = dir + type
                featurefile = svddir + str(week) + '/' + type + featureext
                
                r0weightfile = ilpdir + str(0) + '_' + '_'.join(train) + '_weight_' + "_" + '.json'
                if not fio.IsExist(r0weightfile):#round 0
                    print "first round"
                    firstRound = True

                    Weights = InitializeWeight()
                 
                if not firstRound:
                    print "update weight, round ", round
                    UpdateWeight(BigramIndex, Weights, prefix, L, Lambda, ngram, MalformedFlilter, featurefile)
                
        with open(weightfile, 'w') as fout:
             json.dump(Weights, fout, encoding="utf-8",indent=2)
      
        #fio.SaveDict(BigramIndex, bigramfile)
        
        #fio.SaveDict(Weights, weightfile, True)
        #fio.SaveDict(BigramIndex, bigramfile)

def TestILP(train, test, ilpdir, np, L, Lambda, ngram, MalformedFlilter):
    Weights = {}
    BigramIndex = {}
    
    round = 0
    for round in range(10):
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
            ILP_Supervised(Weights, prefix, L, Lambda, ngram, MalformedFlilter)

def ILP_CrossValidation(ilpdir, np, L, Lambda, ngram, MalformedFlilter, svddir):
    for train, test in LeaveOneLectureOutPermutation():
        TrainILP(train, ilpdir, np, L, Lambda, ngram, MalformedFlilter, svddir)
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
    #ilpdir = "../../data/ILP_Sentence_Supervised_SVD_BOOK/"
    ilpdir = "../../data/ILP_Sentence_Supervised_FeatureWeighting/"
    
    #svddir = "../../data/SVD_Sentence/"
    svddir = ilpdir
    
    corpusname = "book"
    K = 50
       
    MalformedFlilter = False
    ngrams = [1,2]
    
    #for Lambda in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
    for Lambda in [1.0]:
         #for L in [10, 15, 20, 25, 30, 35, 40, 45, 50]:
         for L in [30]:
             for np in ['sentence']: #'chunk
                 ILP_CrossValidation(ilpdir, np, L, Lambda, ngrams, MalformedFlilter, svddir)
    
    print "done"