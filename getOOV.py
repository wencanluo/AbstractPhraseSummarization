from OrigReader import prData
import sys
import re
import fio
import xml.etree.ElementTree as ET
import numpy as np
from collections import defaultdict
import NLTKWrapper
import SennaParser

from Survey import *
import util
                  
def getWordCountDistribution(excelfile, output):
    reload(sys)
    sys.setdefaultencoding('utf8')
    
    header = ['ID', 'Gender', 'Point of Interest', 'Muddiest Point', 'Learning Point']
    summarykey = "Top Answers"
    
    sheets = range(0,12)
    
    datahead = ['Week', '# of sentence POI', '# of sentence POI', '# of sentence POI']
    
    #TA, for the lecture 1 ~ 12
    counts = {'POI':{}, 'MP':{}, 'LP':{}}
    for sheet in sheets:
        row = []
        week = sheet + 1
        
        orig = prData(excelfile, sheet)
        
        for type in ['POI', 'MP', 'LP']:
            summaryList = getTASummary(orig, header, summarykey, type)
            for summary in summaryList:
                unigram = NLTKWrapper.getNgram(summary, 1)
                
                for word in unigram:
                    word = word.lower()
                    if word in NLTKWrapper.punctuations: continue
                    if word not in counts[type]:
                        counts[type][word] = 0
                    counts[type][word] = counts[type][word] + 1
                    
    
    for type in ['POI', 'MP', 'LP']:
        filename = output + "_" + type + '.TA.txt'
        fio.SaveDict(counts[type], filename, True)
    
    #Feedback, from development set
    sheets = range(12,25)  
    counts = {'POI':{}, 'MP':{}, 'LP':{}}
    for sheet in sheets:
        row = []
        week = sheet + 1
        
        orig = prData(excelfile, sheet)
        
        for type in ['POI', 'MP', 'LP']:
            summaryList = getStudentResponseList(orig, header, summarykey, type)
            for summary in summaryList:
                unigram = NLTKWrapper.getNgram(summary, 1)
                
                for word in unigram:
                    word = word.lower()
                    if word in NLTKWrapper.punctuations: continue
                    if word not in counts[type]:
                        counts[type][word] = 0
                    counts[type][word] = counts[type][word] + 1
                    
    
    for type in ['POI', 'MP', 'LP']:
        filename = output + "_" + type + '.Students.txt'
        fio.SaveDict(counts[type], filename, True)

def ExtractNonWord(inputdirpredix, output):
    Models = ["Students"]

    dicts = {}
    
    for model in Models:
        for type in ['POI', 'MP', 'LP']:
            filename = inputdirpredix + "_" + type + "." + model +'.txt'
            dict = fio.LoadDict(filename, 'float')
            dicts[type] = dict

    dict = {}
    for type in ['POI', 'MP', 'LP']:
        util.UnionDict(dict, dicts[type])
    
    wordlist = [word.strip().lower() for word in fio.ReadFile('wordsEn.txt')]
    
    VOC = {}
    for key in dict:
        if key not in wordlist:
            VOC[key] = dict[key]
    
    fio.SaveDict(VOC, output, True)
        
if __name__ == '__main__':
    excelfile = "../../data/2011Spring_norm.xls"
    prefix = '../../data/word_distribution/wd'
    
    getWordCountDistribution(excelfile, prefix)
    ExtractNonWord(prefix, '../../data/voc.txt')
    
    print "done"