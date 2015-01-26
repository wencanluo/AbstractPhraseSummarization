import fio
import json
import sys
import porter
import NLTKWrapper
import os
import json

#Stemming
phraseext = ".key" #a list
studentext = ".keys.source" #json
countext = ".dict"  #a dictionary
lpext = ".lp"
lpsolext = ".sol"

def WriteConstraint1(PhraseBeta, L):
    #$\sum_{j=1}^P y_j \beta _j \le L$
    constraint = []
    for phrase, beta in PhraseBeta.items():
        constraint.append(" ".join([str(beta), phrase]))
    print "  ", " + ".join(constraint), "<=", L

def WriteConstraint2(BigramPhrase):
    #$\sum_{j=1} {y_j Occ_{ij}} \ge x_i$
    for bigram, phrases in BigramPhrase.items():
        print "  ", " + ".join(phrases), "-", bigram, ">=", 0

def WriteConstraint3(PhraseBigram):
    #$y_j Occ_{ij} \le x_i$
    for phrase, bigrams in PhraseBigram.items():
        for bigram in bigrams:
            print "  ", phrase, "-", bigram, "<=", 0
            
def WriteConstraint4(StudentPhrase):
    #$\sum_{j=1}^P {y_j Occ_{jk}} \ge z_k$
    for student, phrases in StudentPhrase.items():
        print "  ", " + ".join(phrases), "-", student, ">=", 0
        
def formulateProblem(BigramTheta, PhraseBeta, BigramPhrase, PhraseBigram, L, lpfileprefix):
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
    
    WriteConstraint2(BigramPhrase)
    
    WriteConstraint3(PhraseBigram)
    
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
    print "Integers"
    print "  ", " ".join(indicators)
    
    #write End
    print "End"
    sys.stdout = SavedStdOut

def SloveILP(lpfileprefix):
    cmd = "gurobi_cl ResultFile=" + lpfileprefix + lpsolext + " " + lpfileprefix + lpext
    os.system(cmd)    

def ExtractSummaryfromILP(lpfileprefix, phrases, output):
    summaries = []
    
    sol = lpfileprefix + lpsolext
    
    lines = fio.ReadFile(sol)
    for line in lines:
        line = line.strip()
        if line.startswith('#'): continue
        
        tokens = line.split()
        assert(len(tokens) == 2)
        
        key = tokens[0]
        value = tokens[1]
        
        if key in phrases:
            if value == '1':
                summaries.append(phrases[key])
    
    fio.SaveList(summaries, output)

def getPhraseBigram(phrasefile):
    #get phrases
    lines = fio.ReadFile(phrasefile)
    phrases = [line.strip() for line in lines]
    
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
    bigramIndex = {}
    i = 1
    for phrase in phrases:
        pKey = phraseIndex[phrase]
        
        #get stemming
        phrase = porter.getStemming(phrase)
        
        #get bigrams
        bigrams = NLTKWrapper.getNgram(phrase, 2)
        for bigram in bigrams:
            if bigram not in bigramIndex:
                bKey = 'X' + str(i)
                bigramIndex[bigram] = bKey
                i = i + 1
            else:
                bKey = bigramIndex[bigram]
            
            PhraseBigram[pKey].append(bKey)
    
    IndexPhrase = {}
    for k,v in phraseIndex.items():
        IndexPhrase[v] = k
    
    IndexBigram = {}
    for k,v in bigramIndex.items():
        IndexBigram[v] = k
        
    return IndexPhrase, IndexBigram, PhraseBigram

def getBigramWeight_TF(PhraseBigram, PhraseIndex, CountFile):
    BigramTheta = {}
    
    CountDict = fio.LoadDict(CountFile, 'float')
    
    for phrase, bigrams in PhraseBigram.items():
        assert(phrase in PhraseIndex)
        p = PhraseIndex[phrase]
        try:
            fequency = CountDict[p]
        except Exception as e:
            print p
            exit()

        for bigram in bigrams:
            if bigram not in BigramTheta:
                BigramTheta[bigram] = 0
            BigramTheta[bigram] = BigramTheta[bigram] + fequency
    
    return BigramTheta

def getWordCounts(phrases):
    PhraseBeta = {}
    for index, phrase in phrases.items():
        N = len(phrase.split())
        PhraseBeta[index] = N
    return PhraseBeta

def getBigramPhrase(PhraseBigram):
    BigramPhrase = {}
    
    for phrase, bigrams in PhraseBigram.items():
        for bigram in bigrams:
            if bigram not in BigramPhrase:
                BigramPhrase[bigram] = []
            BigramPhrase[bigram].append(phrase)
                
    return BigramPhrase

def getStudentPhrase(phrases, sourcefile):
    with open(sourcefile, "r") as infile:
        PhraseStduent = json.load(infile)
    
    indexPhrase = {}
    for index, phrase in phrases.items():
        indexPhrase[phrase] = index
    
    StudentPhrase = {}
    k = 1
    studentIndex = {}
    for phrase, students in PhraseStduent.items():
        pKey = indexPhrase[phrase]
        
        for student in students:
            if student not in studentIndex:
                sKey = 'Z' + str(k)
                studentIndex[student] = sKey
                k = k + 1
            else:
                sKey = studentIndex[student] 
            
            if sKey not in StudentPhrase:
                StudentPhrase[sKey] = []
            StudentPhrase[sKey].append(pKey)
    
    IndexStudent = {}
    for student, index in studentIndex.items():
        IndexStudent[index] = student
        
    return IndexStudent, StudentPhrase

def getStudentWeight_One(StudentPhrase):
    #assume every student is equally important
    StudentGamma = {}
    
    for student in StudentPhrase:
        StudentGamma[student] = 1.0
    return StudentGamma
            
def ILP1(prefix, L):
    # get each stemmed bigram, sequence the bigram and the phrase
    # bigrams: {index:bigram}, a dictionary of bigram index, X
    # phrases: {index:phrase}, is a dictionary of phrase index, Y
    #PhraseBigram: {phrase, [bigram]}
    IndexPhrase, IndexBigram, PhraseBigram = getPhraseBigram(prefix + phraseext)
    fio.SaveDict(IndexPhrase, prefix + ".phrase_index.dict")
    fio.SaveDict(IndexBigram, prefix + ".bigram_index.dict")
    
    #get weight of bigrams
    BigramTheta = getBigramWeight_TF(PhraseBigram, prefix + countext) # return a dictionary
    
    #get word count of phrases
    PhraseBeta = getWordCounts(IndexPhrase)
       
    #get {bigram:[phrase]} dictionary
    BigramPhrase = getBigramPhrase(PhraseBigram)
    
    lpfile = prefix
    formulateProblem(BigramTheta, PhraseBeta, BigramPhrase, PhraseBigram, L, lpfile)
    
    m = SloveILP(lpfile)
    
    output = lpfile + '.L' + str(L) + ".summary"
    ExtractSummaryfromILP(lpfile, IndexPhrase, output)
    
def ILP_Summarizer(ilpdir, np, L):
    sheets = range(0,12)
    
    for i, sheet in enumerate(sheets):
        week = i + 1
        dir = ilpdir + str(week) + '/'
        
        for type in ['POI', 'MP', 'LP']:
            prefix = dir + type + "." + np
            
            ILP1(prefix, L)
            
if __name__ == '__main__':
    ilpdir = "../../data/ILP/"
    
    #ILP1(ilpdir + "test/MP.syntax", 10)
    
    for L in [10, 15, 20, 25, 30, 35, 40, 45, 50]:
        for np in ['syntax', 'chunk']:
            ILP_Summarizer(ilpdir, np, L)

    print "done"