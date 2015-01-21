import sys
import re
import fio
import xml.etree.ElementTree as ET
from collections import defaultdict
from Survey import *
import random
import NLTKWrapper
import SennaParser
import porter
import math
import phrasebasedShallowSummary
import copy
import MaximalMatchTokenizer
import numpy

import ClusterWrapper
import SennaParser

stopwords = [line.lower().strip() for line in fio.readfile("../../../Fall2014/summarization/ROUGE-1.5.5/data/smart_common_words.txt")]
punctuations = ['.', '?', '-', ',', '[', ']', '-', ';', '\'', '"', '+', '&', '!', '/', '>', '<', ')', '(', '#', '=']

def isMalformed(phrase):
    N = len(phrase.split())
    if N == 1: #single stop words
        if phrase.lower() in stopwords: return True
        if phrase.isdigit(): return True
            
    if len(phrase) > 0:
        if phrase[0] in punctuations: return True
    
    return False

def MalformedNPFlilter(NPs):
    newNPs = []
    for NP in NPs:
        if isMalformed(NP): continue
        newNPs.append(NP)
    return newNPs
                        
def getNPs(sennafile, MalformedFlilter=False, source = None, np=None):
    np_phrase = []
    sources = []
    
    #read senna file
    sentences = SennaParser.SennaParse(sennafile)
    
    if len(source) != len(sentences):
        print len(source), len(sentences), sennafile
        
    #get NP phrases
    for i, s in enumerate(sentences):
        if np=='syntax':
            NPs = s.getSyntaxNP()
        else:
            NPs = s.getNPrases()
        
        for NP in NPs:
            NP = NP.lower()
            
            if MalformedFlilter:
                if isMalformed(NP): 
                    #print NP
                    continue
            
            np_phrase.append(NP)
            
            if source != None:
                sources.append(source[i])
    
    if source != None:
        return np_phrase, sources
    
    return np_phrase

def getNPCandiate(student_summaryList, phrasefile, MalformedFlilter=False, source = None, np=None):
    np_phrase = []
    sources = []
    
    #read senna file
    sentences = student_summaryList
    
    if len(source) != len(sentences):
        print len(source), len(sentences), phrasefile
        
    #get NP phrases
    for i, s in enumerate(sentences):
        if np== "candidate":
            NPs = MaximalMatchTokenizer.MaximalMatchTokenizer(s, phrasefile, stemming=False)
        elif np == 'candidatestemming':
            NPs = MaximalMatchTokenizer.MaximalMatchTokenizer(s, phrasefile)
        elif np == "candidatengram":
            NPs = MaximalMatchTokenizer.NgramMatchTokenizer(s, phrasefile, stemming=False)
        elif np == "candidatengramstemming":
            NPs = MaximalMatchTokenizer.NgramMatchTokenizer(s, phrasefile)
        else:
            NPs = []
            
        for NP in NPs:
            NP = NP.lower()
            
            if MalformedFlilter:
                if isMalformed(NP): continue
            
            np_phrase.append(NP)
            
            if source != None:
                sources.append(source[i])
    
    if source != None:
        return np_phrase, sources
    
    return np_phrase

def Similarity2Distance(similarity):
    distance = copy.deepcopy(similarity)
    
    #change the similarity to distance
    for i, row in enumerate(distance):
        for j, col in enumerate(row):
            if distance[i][j] == "NaN":
                distance[i][j] = 1.0
            else:
                try:
                    if float(distance[i][j]) < 0:
                        print "<0", i, j, distance[i][j]
                        distance[i][j] = 0
                    if float(distance[i][j]) > 100:
                        print ">100", i, j, distance[i][j]
                        distance[i][j] = 100
                    distance[i][j] = 1.0 / math.pow(math.e, float(distance[i][j]))
                except OverflowError as e:
                    print e
                    exit()
    return distance

def getPhraseClusterCandidateNP(student_summaryList, weightfile, candiatefile, output, ratio=None, MalformedFlilter=False, source=None, np=None):
    NPCandidates, sources = getNPCandiate(student_summaryList, candiatefile, MalformedFlilter, source=source, np=np)
    
    NPs, matrix = fio.readMatrix(weightfile, hasHead = True)
    
    matrix = Similarity2Distance(matrix)

    index = {}
    for i, NP in enumerate(NPs):
        index[NP] = i
    
    newMatrix = []
    
    for NP1 in NPCandidates:
        assert(NP1 in index)
        i = index[NP1]
        
        row = []
        for NP2 in NPCandidates:
            if NP2 not in index:
                print NP2, weightfile, np
            j = index[NP2]
            row.append(matrix[i][j])
            
        newMatrix.append(row)
    
    V = len(NPCandidates)
    if ratio == "sqrt":
        K = int(math.sqrt(V))
    else:
        K = int(ratio*V)
    
    if K <= 1:
        K = 1
    
    clusterid = ClusterWrapper.KMedoidCluster(newMatrix, K)
 
    body = []   
    for NP, id in zip(NPCandidates, clusterid):
        row = []
        row.append(NP)
        row.append(id)
        body.append(row)    
    
    fio.writeMatrix(output, body, header = None)
    
def getPhraseClusterAll(sennafile, weightfile, output, ratio=None, MalformedFlilter=False, source=None, np=None):
    NPCandidates, sources = getNPs(sennafile, MalformedFlilter, source=source, np=np)
    
    NPs, matrix = fio.readMatrix(weightfile, hasHead = True)
    
    #change the similarity to distance
    matrix = Similarity2Distance(matrix)

    index = {}
    for i, NP in enumerate(NPs):
        index[NP] = i
    
    newMatrix = []
    
    for NP1 in NPCandidates:
        assert(NP1 in index)
        i = index[NP1]
        
        row = []
        for NP2 in NPCandidates:
            if NP2 not in index:
                print NP2, weightfile, np
            j = index[NP2]
            row.append(matrix[i][j])
            
        newMatrix.append(row)
    
    V = len(NPCandidates)
    if ratio == "sqrt":
        K = int(math.sqrt(V))
    elif float(ratio) > 1:
        K = int(ratio)
    else:
        K = int(ratio*V)
    
    if K < 1: K=1
    
    clusterid = ClusterWrapper.KMedoidCluster(newMatrix, K)
    
    body = []   
    for NP, id in zip(NPCandidates, clusterid):
        row = []
        row.append(NP)
        row.append(id)
        body.append(row)    
    
    fio.writeMatrix(output, body, header = None)
    
def getPhraseCluster(phrasedir, method='lexicalOverlapComparer', ratio=None):
    sheets = range(0,12)
    
    for sheet in sheets:
        week = sheet + 1
        for type in ['POI', 'MP', 'LP']:
            weightfilename = phrasedir + str(week)+ '/' + type + '.' + method
            print weightfilename
            
            NPs, matrix = fio.readMatrix(weightfilename, hasHead = True)
            
            #change the similarity to method
            for i, row in enumerate(matrix):
                for j, col in enumerate(row):
                    matrix[i][j] = 1 - float(matrix[i][j]) if matrix[i][j] != "NaN" else 0
            
            V = len(NPs)
            if ratio == None:
                K = int(math.sqrt(V))
            else:
                K = int(ratio*V)
            
            K=10    
            clusterid = ClusterWrapper.KMedoidCluster(matrix, K)
            
#             sorted_lists = sorted(zip(NPs, clusterid), key=lambda x: x[1])
#             NPs, clusterid = [[x[i] for x in sorted_lists] for i in range(2)]
            
            dict = defaultdict(int)
            for id in clusterid:
                dict[id] = dict[id] + 1
             
            body = []   
            for NP, id in zip(NPs, clusterid):
                row = []
                row.append(NP)
                row.append(id)
                #row.append(dict[id])
                
                body.append(row)
            
            if ratio == None:    
                file = phrasedir + '/' + str(week) +'/' + type + ".cluster.kmedoids." + "sqrt" + "." +method
            else:
                file = phrasedir + '/' + str(week) +'/' + type + ".cluster.kmedoids." + str(ratio) + "." +method
            fio.writeMatrix(file, body, header = None)
            
#             dict2 = {}
#             for NP, id in zip(NPs, clusterid):
#                 if id not in dict2:
#                     dict2[id] = []
#                 dict2[id].append(NP)
#             
#             header = ['cluster id', 'NPs', 'count']
#             body = []   
#             for id in set(clusterid):
#                 row = []
#                 row.append(id)
#                 row.append(", ".join(dict2[id]))
#                 row.append(dict[id])
#                 
#                 body.append(row)
#             file = phrasedir + '/' + str(week) +'/' + type + ".cluster.kmedoids.count"
#             fio.writeMatrix(file, body, header)
                
if __name__ == '__main__':
    excelfile = "../data/2011Spring.xls"
    
    sennadatadir = "../data/senna/"
    weigthdir = "../data/np/"
    
    #for method in ['nphard', 'npsoft',  'greedyComparerWNLin']:
#     for method in ['greedyComparerWNLin', 'optimumComparerLSATasa','optimumComparerWNLin',  'dependencyComparerWnLeskTanim', 'bleuComparer', 'cmComparer', 'lsaComparer', 'lexicalOverlapComparer']:
#         datadir = "../../mead/data/ShallowSummary_Weighted" + method + '/'
#         fio.deleteFolder(datadir)
#         ShallowSummary(excelfile, datadir, sennadatadir, K=30, weigthdir=weigthdir, method = method)

    phrasedir = "../data/np/"
    
    #for ratio in [None, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    for ratio in [None]:
        #for method in ['greedyComparerWNLin', 'optimumComparerLSATasa','optimumComparerWNLin',  'dependencyComparerWnLeskTanim', 'lexicalOverlapComparer']: #'bleuComparer', 'cmComparer', 'lsaComparer',
        for method in ['lexicalOverlapComparer']:
            getPhraseCluster(phrasedir, method, ratio)
            print method
    
    print "done"
    
 