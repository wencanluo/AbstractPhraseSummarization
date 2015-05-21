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
    
def formulate_problem(IndexBigram, Weights, PhraseBeta, BigramPhrase, PhraseBigram, partialBigramPhrase, partialPhraseBigram, L, lpfileprefix, FeatureVecU, student_coverage, StudentGamma, StudentPhrase, student_lambda, minthreshold, weight_normalization):
    SavedStdOut = sys.stdout
    sys.stdout = open(lpfileprefix + lpext, 'w')

    #write objective
    print "Maximize"
    objective = []
    
    BigramWeights = ILP_Supervised_FeatureWeight.get_weight_product(Weights, BigramPhrase, IndexBigram, FeatureVecU, minthreshold, weight_normalization)
    #BigramWeights = Weights
    
    if student_coverage:
        for bigram in BigramPhrase:
            if bigram not in BigramWeights: 
                print IndexBigram[bigram]
                continue
            
            w = BigramWeights[bigram]
            
            #if w <= 0: continue
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
    ILP_MC.WriteConstraint1(PhraseBeta, L)
    
    ILP_MC.WriteConstraint2(partialBigramPhrase)
    
    ILP_MC.WriteConstraint3(partialPhraseBigram)
    
    if student_coverage:
        ILP_baseline.WriteConstraint4(StudentPhrase)
           
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
        
def ILP_Supervised(Weights, prefix, featurefile, svdfile, svdpharefile, L, ngram, MalformedFlilter, student_coverage, student_lambda, minthreshold, weight_normalization, sparse_threshold):
    # get each stemmed bigram, sequence the bigram and the phrase
    # bigrams: {index:bigram}, a dictionary of bigram index, X
    # phrases: {index:phrase}, is a dictionary of phrase index, Y
    #PhraseBigram: {phrase, [bigram]}
    IndexPhrase, IndexBigram, PhraseBigram = ILP_baseline.getPhraseBigram(prefix+phraseext, Ngram=ngram, svdfile=svdfile)
    fio.SaveDict(IndexPhrase, prefix + ".phrase_index.dict")
    fio.SaveDict(IndexBigram, prefix + ".bigram_index.dict")
    
    #get weight of bigrams {bigram:weigth}
    #BigramTheta = Weights #ILP.getBigramWeight_TF(PhraseBigram, phrases, prefix + countext) # return a dictionary
    #Weights = ILP_baseline.getBigramWeight_StudentNo(PhraseBigram, IndexPhrase, prefix + countext)
    
    #get word count of phrases
    PhraseBeta = ILP_baseline.getWordCounts(IndexPhrase)
    
    #get {bigram:[phrase]} dictionary
    BigramPhrase = ILP_baseline.getBigramPhrase(PhraseBigram)
    
    partialPhraseBigram, PartialBigramPhrase = ILP_MC.getPartialPhraseBigram(IndexPhrase, IndexBigram, prefix + phraseext, svdfile, svdpharefile, threshold=sparse_threshold)
    fio.SaveDict2Json(partialPhraseBigram, prefix + ".partialPhraseBigram.dict")
    fio.SaveDict2Json(PartialBigramPhrase, prefix + ".PartialBigramPhrase.dict")
    
    #get {student:phrase}
    #sequence students, students = {index:student}
    students, StudentPhrase = ILP_baseline.getStudentPhrase(IndexPhrase, prefix + studentext)
    fio.SaveDict(students, prefix + ".student_index.dict")
    
    #get {student:weight0}
    StudentGamma = ILP_baseline.getStudentWeight_One(StudentPhrase)
    
    FeatureVecU = ILP_Supervised_FeatureWeight.LoadFeatureSet(featurefile)
    
    lpfile = prefix
    formulate_problem(IndexBigram, Weights, PhraseBeta, BigramPhrase, PhraseBigram, PartialBigramPhrase, partialPhraseBigram, L, lpfile, FeatureVecU,
                     student_coverage, StudentGamma, StudentPhrase, student_lambda, minthreshold, weight_normalization)
    
    m = ILP_baseline.SloveILP(lpfile)
    
    output = lpfile + '.L' + str(L) + ".summary"
    ILP_baseline.ExtractSummaryfromILP(lpfile, IndexPhrase, output)

def TestILP(train, test, ilpdir, matrix_dir, np, L, ngram, MalformedFlilter, featuredir, prefixA, student_coverage, student_lambda, minthreshold, weight_normalization, sparse_threshold):
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
            
            svdfile = matrix_dir + str(week) + '/' + type + prefixA
            svdpharefile = matrix_dir + str(week) + '/' + type + '.' + np + ".key"
            
            featurefile = featuredir + str(week) + '/' + type + featureext
            
            ILP_Supervised(Weights, prefix, featurefile, svdfile, svdpharefile, L, ngram, MalformedFlilter, student_coverage, student_lambda, minthreshold, weight_normalization, sparse_threshold)

def ILP_CrossValidation(ilpdir, matrix_dir, np, L, ngram, MalformedFlilter, featuredir, prefixA, student_coverage, student_lambda, minthreshold, weight_normalization, sparse_threshold, no_training):
    for train, test in LeaveOneLectureOutPermutation():
        if not no_training:
            ILP_Supervised_FeatureWeight.TrainILP(train, ilpdir, np, L, student_lambda, ngram, MalformedFlilter, featuredir)
    
    for train, test in LeaveOneLectureOutPermutation():
        TestILP(train, test, ilpdir, matrix_dir, np, L, ngram, MalformedFlilter, featuredir, prefixA, student_coverage, student_lambda, minthreshold, weight_normalization, sparse_threshold)

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
    
    from config import ConfigFile
    
    config = ConfigFile()
    
    matrix_dir = config.get_matrix_dir()
    
    MalformedFlilter = False
    ngrams = config.get_ngrams()
    
    #ILP_baseline.SloveILP(ilpdir + "3/MP.sentence")
    
    #for Lambda in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
    for Lambda in [config.get_student_lambda()]:
        for L in [config.get_length_limit()]:
            for np in ['sentence']: #'chunk
                rank = config.get_rank_max()
                Lambda = config.get_softImpute_lambda()
                if rank == 0:
                    prefixA = '.org.softA'
                else:
                    prefixA = '.' + str(rank) + '_' + str(Lambda) + '.softA'
                    
                for iter in range(config.get_perceptron_maxIter()):
                    ILP_CrossValidation(ilpdir, matrix_dir, np, L, ngrams, MalformedFlilter, featuredir, prefixA=prefixA, 
                                        student_coverage = config.get_student_coverage(), student_lambda = config.get_student_lambda(), 
                                        minthreshold=config.get_perceptron_threshold(), weight_normalization=config.get_weight_normalization(), 
                                        sparse_threshold=config.get_sparse_threshold(), no_training=config.get_no_training())
    
    print "done"