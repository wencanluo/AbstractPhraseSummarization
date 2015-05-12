import fio
import json
import sys
import porter
import NLTKWrapper
import os
import json

import ILP_baseline as ILP

ngramTag = "___"

#Stemming
phraseext = ".key" #a list
studentext = ".keys.source" #json
countext = ".dict"  #a dictionary
lpext = ".lp"
lpsolext = ".sol"

def LoadMC(input):
    with open(input, 'r') as fin:
        A = json.load(fin)
        
    newA = {}
    for bigram, row in A.items():
        bigram = bigram.replace(ngramTag, " ")
        newA[bigram] = row
    return newA

def getNoneZero(A, eps=1e-3):
    nonZero = 0
    N = 0
     
    for bigram, row in A.items():
        N += len(row)
        
        for x in row:
            if abs(x) >= eps:
                nonZero = nonZero + 1
    return nonZero, N
    
def getSparseRatio(svddir, prefixA=".org.softA", eps=1e-3):
    sheets = range(0,12)
    
    total_nonZero = 0.0
    total_N = 0.0
    for sheet in sheets:
        week = sheet + 1
        dir = ilpdir + str(week) + '/'
        
        for type in ['POI', 'MP', 'LP']:
            svdfile = svddir + str(week) + '/' + type + prefixA
        
            A = LoadMC(svdfile)
            
            nonZero, N = getNoneZero(A, eps)
            
            total_nonZero += nonZero
            total_N += N
    
    print total_nonZero, '\t', total_N, '\t', total_nonZero/total_N
           

def WriteConstraint1(PhraseBeta, L):
    #$\sum_{j=1}^P y_j \beta _j \le L$
    constraint = []
    for phrase, beta in PhraseBeta.items():
        constraint.append(" ".join([str(beta), phrase]))
    print "  ", " + ".join(constraint), "<=", L

def WriteConstraint2(partialBigramPhrase):
    #$\sum_{j=1} {y_j Occ_{ij}} \ge x_i$
    for bigram, phrases in partialBigramPhrase.items():
        weightedPhrase = [phrase[1] + ' ' + phrase[0] for phrase in phrases if str(phrase[1]) != '0.0']
        print "  ", " + ".join(weightedPhrase), "-", bigram, ">=", 0

def WriteConstraint3(partialPhraseBigram):
    #$y_j Occ_{ij} \le x_i$
    for phrase, bigrams in partialPhraseBigram.items():
        for bigram in bigrams:
            if str(bigram[1].strip()) == '0.0': continue
            print "  ", bigram[1].strip(), phrase,  "-", bigram[0], "<=", 0
            
def WriteConstraint4(StudentPhrase):
    #$\sum_{j=1}^P {y_j Occ_{jk}} \ge z_k$
    for student, phrases in StudentPhrase.items():
        print "  ", " + ".join(phrases), "-", student, ">=", 0
        
def formulateProblem(BigramTheta, PhraseBeta, partialBigramPhrase, partialPhraseBigram, L, lpfileprefix):
    SavedStdOut = sys.stdout
    sys.stdout = open(lpfileprefix + lpext, 'w')
    
    #write objective
    print "Maximize"
    objective = []
    for bigram, theta in BigramTheta.items():
        objective.append(" ".join([str(theta), bigram]))
    print "  ", " + ".join(objective)
    
    #write constraints
    print "Subject To"
    WriteConstraint1(PhraseBeta, L)
    
    WriteConstraint2(partialBigramPhrase)
    
    WriteConstraint3(partialPhraseBigram)
    
    indicators = []
    for bigram in BigramTheta.keys():
        indicators.append(bigram)
    for phrase in PhraseBeta.keys():
        indicators.append(phrase)
        
    #write Bounds
    print "Bounds"
    for indicator in indicators:
        print "  ", indicator, "<=", 1
    
    #write Integers
    indicators = []
#     for bigram in BigramTheta.keys():
#         indicators.append(bigram)
    for phrase in PhraseBeta.keys():
        indicators.append(phrase)
        
    print "Integers"
    print "  ", " ".join(indicators)
    
    #write End
    print "End"
    sys.stdout = SavedStdOut

def getBigramKey(bigram):
    bigram = bigram.replace(" ", ngramTag)
    bigramKey = porter.getStemming(bigram)
    return bigramKey

# Default bigram dict: stemmed bigram individual
# SVD bigram dict: bigram tag, + stemmed
def getRecoveredKeyDict(phrasefile):
    phrases, bigrams_org, PhraseBigram = ILP.getPhraseBigram(phrasefile, Ngram=[1,2], NoStopWords=True, Stemmed=False)
    
    dict = {}
    for index, bigram in bigrams_org.items():
        default_bigram = porter.getStemming(bigram)
        svd_bigram = getBigramKey(bigram)
        
        dict[svd_bigram] = default_bigram
    return dict
    
def getPartialPhraseBigram(IndexPhrase, IndexBigram, phrasefile, svdfile, svdpharefile, threshold=0.5):
    lines = fio.ReadFile(phrasefile)
    phrases = [line.strip() for line in lines]
    
    #bigramKeyDict = getRecoveredKeyDict(phrasefile)
    
    lines = fio.ReadFile(svdpharefile)
    svdphrases = [line.strip() for line in lines]
    
    svdA = LoadMC(svdfile)
    
    BigramIndex = {}
    for k,v in IndexBigram.items():
        BigramIndex[v] = k
    
    PhraseIndex = {}
    for k,v in IndexPhrase.items():
        PhraseIndex[v] = k
        
    PartialPhraseBigram = {}
    for phrase in phrases:
        if phrase not in PhraseIndex:
            print "phrase not in PhraseIndex:", phrase
            continue
        i = svdphrases.index(phrase)
        if i==-1:
            print phrase
            continue
        
        pKey = PhraseIndex[phrase]
        if pKey not in PartialPhraseBigram:
            PartialPhraseBigram[pKey] = []
        
        for bigram in svdA.keys():
            row = svdA[bigram]
            svdvalue = row[i]
            
            if bigram not in BigramIndex: 
                print "bigram not in BigramIndex:", bigram
                continue
            bKey = BigramIndex[bigram]
            
            if row[i] < threshold: continue
            #print bigram, phrase, svdvalue
            #if str(row[i]) == '0.0': continue
            PartialPhraseBigram[pKey].append([bKey, str(row[i])])
    
    PartialBigramPhrase = {}
    for bigram in svdA.keys():       
        row = svdA[bigram]
        
        bigram = bigram.replace(ngramTag, " ")
        if bigram not in BigramIndex: 
            continue
        bKey = BigramIndex[bigram]
        
        if bKey not in PartialBigramPhrase:
            PartialBigramPhrase[bKey] = []
            
        for phrase in phrases:
            if phrase not in PhraseIndex:
                print phrase
                continue
            
            i = svdphrases.index(phrase)
            if i==-1:
                print phrase
                continue
        
            pKey = PhraseIndex[phrase]
            
            if row[i] < threshold: continue
            #if str(row[i]) == '0.0': continue
            PartialBigramPhrase[bKey].append([pKey, str(row[i])])
    
    return PartialPhraseBigram, PartialBigramPhrase
    
def ILP1(prefix, svdfile, svdpharefile, L, threshold=0.5):
    # get each stemmed bigram, sequence the bigram and the phrase
    # bigrams: {index:bigram}, a dictionary of bigram index, X
    # phrases: {index:phrase}, is a dictionary of phrase index, Y
    #PhraseBigram: {phrase, [bigram]}
    IndexPhrase, IndexBigram, PhraseBigram = ILP.getPhraseBigram(prefix + phraseext, Ngram=[1,2], svdfile=svdfile)
    fio.SaveDict(IndexPhrase, prefix + ".phrase_index.dict")
    fio.SaveDict(IndexBigram, prefix + ".bigram_index.dict")
    
    #get weight of bigrams
    BigramTheta = ILP.getBigramWeight_StudentNo(PhraseBigram, IndexPhrase, prefix + countext) # return a dictionary
    fio.SaveDict(BigramTheta, prefix + ".bigram_theta.dict")
    
    #get word count of phrases
    PhraseBeta = ILP.getWordCounts(IndexPhrase)
    
    #getPartial Bigram Phrase matrix
    partialPhraseBigram, PartialBigramPhrase = getPartialPhraseBigram(IndexPhrase, IndexBigram, prefix + phraseext, svdfile, svdpharefile, threshold=threshold)
       
    #get {bigram:[phrase]} dictionary
    BigramPhrase = ILP.getBigramPhrase(PhraseBigram)
    
    lpfile = prefix
    formulateProblem(BigramTheta, PhraseBeta, PartialBigramPhrase, partialPhraseBigram, L, lpfile)
    
    m = ILP.SloveILP(lpfile)
    
    output = lpfile + '.L' + str(L) + ".summary"
    ILP.ExtractSummaryfromILP(lpfile, IndexPhrase, output)
    
def ILP_Summarizer(ilpdir, svddir, np, L, prefixA=".org.softA", threshold=0.5):
    sheets = range(0,12)
    #sheets = range(2,3)
    
    for sheet in sheets:
        week = sheet + 1
        dir = ilpdir + str(week) + '/'
        
        for type in ['POI', 'MP', 'LP']:
            prefix = dir + type + "." + np
            svdfile = svddir + str(week) + '/' + type + prefixA
            svdpharefile = svddir + str(week) + '/' + type + '.' + np + ".key"
            print prefix
            print svdfile
                
            ILP1(prefix, svdfile, svdpharefile, L, threshold=threshold)
            
if __name__ == '__main__':
    ilpdir = "../../data/ILP1_Sentence_MC/"
    svddir = "../../data/SVD_Sentence/"
    #ILP1(ilpdir + "test/MP.syntax", 10)
    
#     for L in [10, 15, 20, 25, 30, 35, 40, 45, 50]:
#         for np in ['syntax', 'chunk']:
#             ILP_Summarizer(ilpdir, np, L)
#     SloveILP("../../data/ILP1_Sentence_SVD/3/MP.sentence")
#    
#     A = {'a':[1,0], 'b':[0,0,1]}
#     A = LoadMC("../../data/SVD_Sentence/3/MP.org.softA")
#     print getNoneZero(A)
#     for eps in [1e-4, 1e-3, 1e-2, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
#         print eps, '\t',
#         getSparseRatio(svddir, prefixA=".200_2.softA", eps=eps)
#     exit()
    
    for L in [30]:
        for np in ['sentence']:
            ILP_Summarizer(ilpdir, svddir, np, L, prefixA=".200_2.softA", threshold=-100) #".2690_0.1.softA"
            
    print "done"