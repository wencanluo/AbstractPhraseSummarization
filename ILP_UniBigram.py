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
    
def formulate_problem(Lambda, StudentGamma, StudentPhrase, BigramTheta, PhraseBeta, BigramPhrase, PhraseBigram, L, lpfileprefix):
    SavedStdOut = sys.stdout
    sys.stdout = open(lpfileprefix + lpext, 'w')
    
    #write objective
    print "Maximize"
    objective = []
    for bigram, theta in BigramTheta.items():
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
        
def ILP2(prefix, L, Lambda, ngram, MalformedFlilter):
    # get each stemmed bigram, sequence the bigram and the phrase
    # bigrams: {index:bigram}, a dictionary of bigram index, X
    # phrases: {index:phrase}, is a dictionary of phrase index, Y
    #PhraseBigram: {phrase, [bigram]}
    phrases, bigrams, PhraseBigram = ILP.getPhraseBigram(prefix + phraseext, Ngram=ngram, MalformedFlilter=MalformedFlilter)
    fio.SaveDict(phrases, prefix + ".phrase_index.dict")
    fio.SaveDict(bigrams, prefix + ".bigram_index.dict")
    
    #get weight of bigrams {bigram:weigth}
    BigramTheta = ILP.getBigramWeight_TF(PhraseBigram, phrases, prefix + countext) # return a dictionary
    
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
    formulate_problem(Lambda, StudentGamma, StudentPhrase, BigramTheta, PhraseBeta, BigramPhrase, PhraseBigram, L, lpfile)
    
    m = ILP.SloveILP(lpfile)
    
    output = lpfile + '.L' + str(L) + "." + str(Lambda) + ".summary"
    ILP.ExtractSummaryfromILP(lpfile, phrases, output)
    
def ILP_Summarizer(ilpdir, np, L, Lambda, ngram, MalformedFlilter):
    sheets = range(0,12)
    
    for i, sheet in enumerate(sheets):
        week = i + 1
        dir = ilpdir + str(week) + '/'
        
        for type in ['POI', 'MP', 'LP']:
            prefix = dir + type + "." + np
            
            ILP2(prefix, L, Lambda, ngram, MalformedFlilter)
            
if __name__ == '__main__':
    
#     ilpdir = "../../data/ILP2/"
#      
#     #for Lambda in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
#     for Lambda in [0.8]:
#         #for L in [10, 15, 20, 25, 30, 35, 40, 45, 50]:
#         for L in [30]:
#             for np in ['syntax', ]: #'chunk'
#                 ILP_Summarizer(ilpdir, np, L, Lambda, ngram=[2], MalformedFlilter=False)
#                    
#     ilpdir = "../../data/ILP_Unibigram/"
#      
#     #for Lambda in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
#     for Lambda in [0.8]:
#         #for L in [10, 15, 20, 25, 30, 35, 40, 45, 50]:
#         for L in [30]:
#             for np in ['syntax', ]: #'chunk'
#                 ILP_Summarizer(ilpdir, np, L, Lambda, ngram=[1,2], MalformedFlilter=False)
#      
#                     
#     ilpdir = "../../data/ILP_UnibigramMalformedFilter/"
#     
#     #for Lambda in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
#     for Lambda in [0.8]:
#         #for L in [10, 15, 20, 25, 30, 35, 40, 45, 50]:
#         for L in [30]:
#             for np in ['syntax', ]: #'chunk'
#                 ILP_Summarizer(ilpdir, np, L, Lambda, ngram=[1,2], MalformedFlilter=True)   

    ilpdir = "../../data/ILP2_MalformedFilter/"
     
    #for Lambda in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
    for Lambda in [0.8]:
        #for L in [10, 15, 20, 25, 30, 35, 40, 45, 50]:
        for L in [30]:
            for np in ['syntax', ]: #'chunk'
                ILP_Summarizer(ilpdir, np, L, Lambda, ngram=[2], MalformedFlilter=True) 

    print "done"