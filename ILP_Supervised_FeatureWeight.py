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
#from ltpservice.LTPOption import POS

maxIter = 100

#Stemming
phraseext = ".key" #a list
studentext = ".keys.source" #json
countext = ".dict"  #a dictionary
lpext = ".lp"
lpsolext = ".sol"
sumexe = ".ref.summary"
featureext = ".f"

ngramTag = "___"

def LoadFeatureSet(featurename):
    with open(featurename, 'r') as fin:
        featureV = json.load(fin)
        
    return featureV

def get_weight_product(Weights, BigramPhrase, IndexBigram, FeatureVecU, minthreshold, weight_normalization):
    BigramWeights = {}
    
    for bigram in BigramPhrase:
        bigramname = IndexBigram[bigram]
                
        if bigramname in FeatureVecU:
            fvec = FeatureVector(FeatureVecU[bigramname])
            w = Weights.dot(fvec)
            BigramWeights[bigram] = w
    
    median_w = numpy.median(BigramWeights.values())
    mean_w = numpy.mean(BigramWeights.values())
    std_w = numpy.std(BigramWeights.values())
    max_w = numpy.max(BigramWeights.values())
    min_w = numpy.min(BigramWeights.values())
                    
    if weight_normalization == 0:
        for bigram in BigramWeights:
            w = BigramWeights[bigram]
            BigramWeights[bigram] = w - minthreshold            
    elif weight_normalization == 1:#normalize to 0 ~ 1
        for bigram in BigramWeights:
            w = BigramWeights[bigram]
            if (max_w - min_w) != 0:
                BigramWeights[bigram] = (w - min_w)/(max_w - min_w)
    elif weight_normalization == 2:#normalize to 0 ~ 1
        for bigram in BigramWeights:
            w = BigramWeights[bigram]
            if (max_w - mean_w - std_w) != 0:
                BigramWeights[bigram] = (w - mean_w - std_w)/(max_w - mean_w - std_w)
    else:
        pass
                
    return BigramWeights

def formulate_problem(IndexBigram, Weights, PhraseBeta, BigramPhrase, PhraseBigram, L, lpfileprefix, FeatureVecU, student_coverage, StudentGamma, StudentPhrase, student_lambda, minthreshold, weight_normalization):
    SavedStdOut = sys.stdout
    sys.stdout = open(lpfileprefix + lpext, 'w')

    #write objective
    print "Maximize"
    objective = []
    
    BigramWeights = get_weight_product(Weights, BigramPhrase, IndexBigram, FeatureVecU, minthreshold, weight_normalization)
    
    if os.name == 'nt':
        import matplotlib.pyplot as plt
        plt.clf()
        plt.hist(BigramWeights.values(), bins=50)
        plt.savefig(lpfileprefix + '.png')
        fio.SaveDict(BigramWeights, lpfileprefix + '.bigram_weight.txt', SortbyValueflag=True)
    
    if student_coverage:
        for bigram in BigramPhrase:
            if bigram not in BigramWeights: 
                print IndexBigram[bigram]
                continue
            
            w = BigramWeights[bigram]
            
            if w <= 0: continue
            objective.append(" ".join([str(w*student_lambda), bigram]))
                     
        for student, grama in StudentGamma.items():
            if Lambda==1:continue
             
            objective.append(" ".join([str(grama*(1-student_lambda)), student]))
    else:
        for bigram in BigramPhrase:
            if bigram not in BigramWeights: continue
            bigramname = IndexBigram[bigram]
                    
            w = BigramWeights[bigram]
            if w <= 0: continue
            objective.append(" ".join([str(w), bigram]))
    
    print "  ", " + ".join(objective)
    
    #write constraints
    print "Subject To"
    ILP.WriteConstraint1(PhraseBeta, L)
    
    ILP.WriteConstraint2(BigramPhrase)
    
    ILP.WriteConstraint3(PhraseBigram)
    
    if student_coverage:
        ILP.WriteConstraint4(StudentPhrase)
    
    indicators = []
    for bigram in BigramPhrase.keys():
        indicators.append(bigram)
    for phrase in PhraseBeta.keys():
        indicators.append(phrase)
        
    if student_coverage:
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
        
def ILP_Supervised(Weights, prefix, featurefile, L, ngram, MalformedFlilter, student_coverage, student_lambda, minthreshold, weight_normalization):
    # get each stemmed bigram, sequence the bigram and the phrase
    # bigrams: {index:bigram}, a dictionary of bigram index, X
    # phrases: {index:phrase}, is a dictionary of phrase index, Y
    #PhraseBigram: {phrase, [bigram]}
    IndexPhrase, IndexBigram, PhraseBigram = ILP.getPhraseBigram(prefix+phraseext, Ngram=ngram)
    fio.SaveDict(IndexPhrase, prefix + ".phrase_index.dict")
    fio.SaveDict(IndexBigram, prefix + ".bigram_index.dict")
    
    #get weight of bigrams {bigram:weigth}
    #BigramTheta = Weights #ILP.getBigramWeight_TF(PhraseBigram, phrases, prefix + countext) # return a dictionary
    
    #get word count of phrases
    PhraseBeta = ILP.getWordCounts(IndexPhrase)
    
    #get {bigram:[phrase]} dictionary
    BigramPhrase = ILP.getBigramPhrase(PhraseBigram)

    #get {student:phrase}
    #sequence students, students = {index:student}
    students, StudentPhrase = ILP.getStudentPhrase(IndexPhrase, prefix + studentext)
    fio.SaveDict(students, prefix + ".student_index.dict")
    
    #get {student:weight0}
    StudentGamma = ILP.getStudentWeight_One(StudentPhrase)
    
    FeatureVecU = LoadFeatureSet(featurefile)
    
    lpfile = prefix
    formulate_problem(IndexBigram, Weights, PhraseBeta, BigramPhrase, PhraseBigram, L, lpfile, FeatureVecU, 
                     student_coverage, StudentGamma, StudentPhrase, student_lambda, minthreshold, weight_normalization)
    
    m = ILP.SloveILP(lpfile)
    
    output = lpfile + '.L' + str(L) + ".summary"
    ILP.ExtractSummaryfromILP(lpfile, IndexPhrase, output)

def getLastIndex(BigramIndex):
    maxI = 1
    for bigram in BigramIndex.values():
        if int(bigram[1:]) > maxI:
            maxI = int(bigram[1:])
    return maxI

def initialize_weight():
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
            length -= len(line.split())
    
    fio.SaveList(summaries, sumfile)

def preceptron_update(Weights, prefix, L, Lambda, ngram, MalformedFlilter, featurefile):
    #scan all the bigrams in the responses
    _, IndexBigram, SummaryBigram = ILP.getPhraseBigram(prefix+phraseext, Ngram=ngram, MalformedFlilter=MalformedFlilter)
    
    reffile = ExtractRefSummaryPrefix(prefix) + '.ref.summary'
    _, IndexRefBigram, SummaryRefBigram = ILP.getPhraseBigram(reffile, Ngram=ngram, MalformedFlilter=MalformedFlilter)
    
    RefBigramDict = getBigramDict(IndexRefBigram, SummaryRefBigram)
    
    #update the weights
    FeatureVecU = LoadFeatureSet(featurefile)
    
    #i = getLastIndex(BigramIndex)
        
    pos = 0
    neg = 0
    correct_pos = 0
    correct_neg = 0
    
    for summary, bigrams in SummaryBigram.items():
        for bigram in bigrams:
            bigramname = IndexBigram[bigram]
            if bigramname not in FeatureVecU: 
                print bigramname
                continue
            
            vec = FeatureVector(FeatureVecU[bigramname])
            y = 1.0 if bigramname in RefBigramDict else -1.0
            
            if Weights.dot(vec)*y <= 0:
                Weights += y*vec
                
                if y==1.0:
                    pos += 1
                else:
                    neg += 1
            else:
                if y==1.0:
                    correct_pos += 1
                else:
                    correct_neg += 1
    
    print "pos:", pos
    print "neg:", neg
    print "correct_pos:", correct_pos
    print "correct_neg:", correct_neg
        
    return Weights

def UpdateWeight_iterate(BigramIndex, Weights, prefix, L, Lambda, ngram, MalformedFlilter, featurefile):
    ILP_Supervised(Weights, prefix, featurefile, L, Lambda, ngram, MalformedFlilter)
    
    #read the summary, update the weight 
    sumfile = prefix + '.L' + str(L) + "." + str(Lambda) + '.summary'
    
    if len(fio.ReadFile(sumfile)) == 0:#no summary is generated, using a random baseline      
        generate_randomsummary(prefix, L, sumfile)
    
    if len(fio.ReadFile(sumfile)) == 0:
        debug = 1
        
    _, IndexBigram, SummaryBigram = ILP.getPhraseBigram(sumfile, Ngram=ngram, MalformedFlilter=MalformedFlilter)
    
    reffile = ExtractRefSummaryPrefix(prefix) + '.ref.summary'
    _, IndexRefBigram, SummaryRefBigram = ILP.getPhraseBigram(reffile, Ngram=ngram, MalformedFlilter=MalformedFlilter)
    
    RefBigramDict = getBigramDict(IndexRefBigram, SummaryRefBigram)
    
    #update the weights
    FeatureVecU = LoadFeatureSet(featurefile)
    
    i = getLastIndex(BigramIndex)
    
    #if the generated summary matches the golden summary, update the bigrams
    
    #get the bigrams
    #{sentence:bigrams}
    
    pos = 0
    neg = 0
    correct_pos = 0
    correct_neg = 0
    
    for summary, bigrams in SummaryBigram.items():
        for bigram in bigrams:
            bigramname = IndexBigram[bigram]
            if bigramname not in FeatureVecU: 
                print bigramname
                continue
            
            vec = FeatureVector(FeatureVecU[bigramname])
            y = 1.0 if bigramname in RefBigramDict else -1.0
            
            if Weights.dot(vec)*y <= 0:
                Weights += y*vec
                
                if y==1.0:
                    pos += 1
                else:
                    neg += 1
            else:
                if y==1.0:
                    correct_pos += 1
                else:
                    correct_neg += 1
    
    print "pos:", pos
    print "neg:", neg
    print "correct_pos:", correct_pos
    print "correct_neg:", correct_neg
        
    return Weights

def UpdateWeight_old(BigramIndex, Weights, prefix, L, Lambda, ngram, MalformedFlilter, featurefile):
    ILP_Supervised(Weights, prefix, featurefile, L, Lambda, ngram, MalformedFlilter)
    
    #read the summary, update the weight 
    sumfile = prefix + '.L' + str(L) + "." + str(Lambda) + '.summary'
    
    if len(fio.ReadFile(sumfile)) == 0:#no summary is generated, using a random baseline      
        generate_randomsummary(prefix, L, sumfile)
    
    if len(fio.ReadFile(sumfile)) == 0:
        debug = 1
        
    _, IndexBigram, SummaryBigram = ILP.getPhraseBigram(sumfile, Ngram=ngram, MalformedFlilter=MalformedFlilter)
    
    reffile = ExtractRefSummaryPrefix(prefix) + '.ref.summary'
    _, IndexRefBigram, SummaryRefBigram = ILP.getPhraseBigram(reffile, Ngram=ngram, MalformedFlilter=MalformedFlilter)
    
    RefBigramDict = getBigramDict(IndexRefBigram, SummaryRefBigram)
    
    #update the weights
    FeatureVecU = LoadFeatureSet(featurefile)
    
    i = getLastIndex(BigramIndex)
    
    #if the generated summary matches the golden summary, update the bigrams
    
    #get the bigrams
    #{sentence:bigrams}
    
    positve = []
    negative = []
    for summary, bigrams in SummaryBigram.items():
        for bigram in bigrams:
            bigramname = IndexBigram[bigram]
            if bigramname not in FeatureVecU: 
                print bigramname
                continue
            
            if bigramname in RefBigramDict:
                positve.append(bigramname)
            else:
                #Weights = Weights.sub_cutoff(vec)
                negative.append(bigramname)
    
    #get the feature set
    positive_feature_set = FeatureVector()
    for bigram in positve:
        vec = FeatureVector(FeatureVecU[bigram])
        positive_feature_set += vec
    for k, v in positive_feature_set.iteritems():
        positive_feature_set[k] = 1.0
    
    negative_feature_set = FeatureVector()
    for bigram in negative:
        vec = FeatureVector(FeatureVecU[bigram])
        negative_feature_set += vec
    for k, v in negative_feature_set.iteritems():
        negative_feature_set[k] = 1.0
    
    print "positive", len(positive_feature_set)
    print "negative", len(negative_feature_set)
    
    if len(negative_feature_set) == 0:
        debug = 1
    
    #feature update
    
    Weights += positive_feature_set
    #Weights -= negative_feature_set
    
    return Weights

def TrainILP(train, ilpdir, np, L, Lambda, ngram, MalformedFlilter, featuredir):
    Weights = {} #{Index:Weight}
    #BigramIndex = {} #{bigram:index}
    
    round = 0
    for round in range(maxIter):
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
                featurefile = featuredir + str(week) + '/' + type + featureext
                
                r0weightfile = ilpdir + str(0) + '_' + '_'.join(train) + '_weight_' + "_" + '.json'
                if not fio.IsExist(r0weightfile):#round 0
                    print "first round"
                    firstRound = True

                    Weights = initialize_weight()
                 
                if not firstRound:
                    print "update weight, round ", round
                    preceptron_update(Weights, prefix, L, Lambda, ngram, MalformedFlilter, featurefile)
                
        with open(weightfile, 'w') as fout:
             json.dump(Weights, fout, encoding="utf-8",indent=2)
      
        #fio.SaveDict(BigramIndex, bigramfile)
        
        #fio.SaveDict(Weights, weightfile, True)
        #fio.SaveDict(BigramIndex, bigramfile)

def TestILP(train, test, ilpdir, np, L, ngram, MalformedFlilter, featuredir, student_coverage, student_lambda, minthreshold, weight_normalization):
    Weights = {}
    
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
        
    for sheet in test:
        week = int(sheet) + 1
        dir = ilpdir + str(week) + '/'
        
        for type in ['POI', 'MP', 'LP']:
            prefix = dir + type + "." + np
            print "Test: ", prefix
            
            featurefile = featuredir + str(week) + '/' + type + featureext
            ILP_Supervised(Weights, prefix, featurefile, L, ngram, MalformedFlilter, student_coverage, student_lambda, minthreshold, weight_normalization)

def ILP_CrossValidation(ilpdir, np, L, ngram, MalformedFlilter, featuredir, student_coverage, student_lambda, minthreshold, weight_normalization, no_training):
    for train, test in LeaveOneLectureOutPermutation():
        if not no_training:
            TrainILP(train, ilpdir, np, L, Lambda, ngram, MalformedFlilter, featuredir)
    for train, test in LeaveOneLectureOutPermutation():
        TestILP(train, test, ilpdir, np, L, ngram, MalformedFlilter, featuredir, student_coverage, student_lambda, minthreshold, weight_normalization)

def LeaveOneLectureOutPermutation():
    sheets = range(0,12)
    N = len(sheets)
    for i in range(N):
        train = [str(k) for k in range(N) if k != i]
        #train = [str(i)]
        test = [str(i)]
        yield train, test
            
if __name__ == '__main__':   
    
    from config import ConfigFile
    config = ConfigFile()
    
    ilpdir = "../../data/ILP_Sentence_Supervised_FeatureWeighting_Normalization/"
    
    featuredir = ilpdir
    
    MalformedFlilter = False
    ngrams = config.get_ngrams()
    
    #for Lambda in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
    for Lambda in [config.get_student_lambda()]:
         #for L in [10, 15, 20, 25, 30, 35, 40, 45, 50]:
         for L in [config.get_length_limit()]:
             for np in ['sentence_filter']: #'chunk\
                 for iter in range(config.get_perceptron_maxIter()):
                     ILP_CrossValidation(ilpdir, np, L, ngrams, MalformedFlilter, featuredir, 
                                         student_coverage = config.get_student_coverage(), 
                                         student_lambda = config.get_student_lambda(), 
                                         minthreshold=config.get_perceptron_threshold(), 
                                         weight_normalization=config.get_weight_normalization(), no_training=config.get_no_training())
    
    print "done"