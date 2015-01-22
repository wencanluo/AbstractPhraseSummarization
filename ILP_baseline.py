import fio
import json
from gurobipy import *

#Stemming
phraseext = ".key" #a list
studentext = ".source" #json
countext = ".dict"  #a dictionary

def formulateProblem(input, lpfile):
    #write objective
    print "Maximize"
    
    #write constraints
    print "Subject To"
    
    #write Bounds
    print "Bounds"
    
    #write Integers
    print "Integers"
    
    #write End
    print "End"

def SolveProblem():
    pass

def ILP1(prefix, L):
    PhraseBigram = getPhraseBigram(prefix + phraseext) #X
    
    Bigram = getBigram(PhraseBigram)
    
    Theta = getBigramWeight_TF(PhraseBigram, prefix + countext) #Theta

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
    
    ILP1(ilpdir + "test/MP.syntax")
    
#     for L in [10, 15, 20, 25, 30, 35, 40, 45, 50]:
#         for np in ['syntax', 'chunk']:
#             ILP_Summarizer(ilpdir, np, L)