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

import tfidf
import shallowSummary
import phraseClusteringKmedoid


stopwords = [line.lower().strip() for line in fio.readfile(stopwordfilename)]
print "stopwords:", len(stopwords)

stopwords = stopwords + ['.', '?', '-', ',', '[', ']', '-', ';', '\'', '"', '+', '&', '!', '/', '>', '<', ')', '(', '#', '=']


def getOverlap(dict1, dict2):
    count = 0
    for key in dict1.keys():
        if key in stopwords: 
            continue
        if key in dict2:
            count = count + 1
    return count

def getStemDict(words):
    dict = {}
    stemed = porter.getStemming(words)
    for token in stemed.split():
        dict[token] = 1
    return dict
                    
def getKeyNgram(student_summaryList, sennafile, save2file=None, soft = False):
    np_phrase = defaultdict(float)
    
    #read senna file
    sentences = SennaParser.SennaParse(sennafile)
    
    stemdict = {}
    
    #get NP phrases
    for s in sentences:
        NPs = s.getNPrases()
        
        for NP in NPs:
            NP = NP.lower()
            
            if soft:
                #cache the stem dictionary
                if NP not in stemdict:
                    stemdict[NP] = getStemDict(NP)
                
                print "----------------------------------"
                print "current dict:"
                fio.PrintDict(np_phrase)
                print "new phrase:" + NP
                
                #update count
                duplicateFlag = False
                for key in np_phrase.keys():
                    overlap_count = getOverlap(stemdict[key], stemdict[NP])
                    if overlap_count >= 1:
                        duplicateFlag = True
                        if NP != key:
                            np_phrase[NP] = np_phrase[NP] + overlap_count
                            np_phrase[key] = np_phrase[key] + overlap_count
                        else:
                            np_phrase[key] = np_phrase[key] + overlap_count
                
                if not duplicateFlag:
                    np_phrase[NP] = np_phrase[NP] + 1
                
            else:
                np_phrase[NP] = np_phrase[NP] + 1
    
    if save2file != None:
        fio.SaveDict(np_phrase, save2file, SortbyValueflag = True)
        
    return np_phrase
            
def getKeyPhrases(student_summaryList, sennafile, save2file=None):
    return getKeyNgram(student_summaryList, sennafile, save2file=save2file)
                            
def getShallowSummary(excelfile, folder, sennadatadir, tfidfdir, np, method, K=30):
    #K is the number of words per points
    header = ['ID', 'Gender', 'Point of Interest', 'Muddiest Point', 'Learning Point']
    summarykey = "Top Answers"
    
    #sheets = range(0,25)
    sheets = range(0,12)
    
    for i, sheet in enumerate(sheets):
        week = i + 1
        orig = prData(excelfile, sheet)
        
        for type in ['POI', 'MP', 'LP']:
            print excelfile, sheet, type
            student_summaryList = getStudentResponseList(orig, header, summarykey, type)

            path = folder + str(week)+ '/'
            fio.newPath(path)
            filename = path + type + '.summary'
            
            sennafile = sennadatadir + "senna." + str(week) + "." + type + '.output'
            
            Summary = []
            
            if method == 'tfidf':
                dict = fio.LoadDict(tfidfdir + str(week)+ '/' + type + '.' + np + '.tfidf.dict')
            if method.startswith('lexrank'):
                dict = fio.LoadDict(tfidfdir + str(week)+ '/' + type + '.' + np + '.'+method+'.dict')
            else:
                dict = getKeyPhrases(student_summaryList, sennafile, save2file=filename + ".keys")
            
            keys = sorted(dict, key=dict.get, reverse = True)
            
            total_word = 0
            word_count = 0
            for key in keys:
                skip = False
                for s in Summary:
                    if getOverlap(getStemDict(s), getStemDict(key)) > 0: #duplicate removing
                        skip = True
                        continue
                if skip: continue
                word_count = len(key.split())
                total_word = total_word + word_count
                
                if len(Summary) + 1 <= K:
                #if total_word <= K:
                    Summary.append(key)
            
            fio.savelist(Summary, filename)
                        
def ShallowSummary(excelfile, datadir, sennadatadir, tfidfdir, np, method, K=30):
    getShallowSummary(excelfile, datadir, sennadatadir, tfidfdir, np, method,  K)
    WriteTASummary(excelfile, datadir)

def computeTFIDF(excelfile, datadir, sennadatadir, np):
    header = ['ID', 'Gender', 'Point of Interest', 'Muddiest Point', 'Learning Point']
    summarykey = "Top Answers"
    
    for type in ['POI', 'MP', 'LP']:
        my_tfidf = tfidf.TfIdf(stopword_filename=stopwordfilename)
        
        sheets = range(0,12)
        for i, sheet in enumerate(sheets):
            week = i + 1
            orig = prData(excelfile, sheet)
            student_summaryList = getStudentResponseList(orig, header, summarykey, type, withSource=True)
            ids = [summary[1] for summary in student_summaryList]
            summaries = [summary[0] for summary in student_summaryList] 
            
            sennafile = sennadatadir + "senna." + str(week) + "." + type + '.output'
            NPs, sources = phraseClusteringKmedoid.getNPs(sennafile, MalformedFlilter = True, source=ids, np=np)
            
            my_tfidf.add_input_document_withterms(NPs)
        
        for i, sheet in enumerate(sheets):
            week = i + 1        
            
            orig = prData(excelfile, sheet)
            student_summaryList = getStudentResponseList(orig, header, summarykey, type, withSource=True)
            ids = [summary[1] for summary in student_summaryList]
            summaries = [summary[0] for summary in student_summaryList] 
            
            sennafile = sennadatadir + "senna." + str(week) + "." + type + '.output'
            NPs, sources = phraseClusteringKmedoid.getNPs(sennafile, MalformedFlilter = True, source=ids, np=np)
            
            dict = my_tfidf.get_doc_keywords_withterms(NPs)
            
            path = datadir + str(week)+ '/'
            fio.newPath(path)
            fio.SaveDict(dict, path + type + '.' + np + '.tfidf.dict', SortbyValueflag = True)

def getTFIDF(excelfile, datadir, sennadatadir, np):
    computeTFIDF(excelfile, datadir, sennadatadir, np)
            
if __name__ == '__main__':
    excelfile = "../data/2011Spring.xls"
    
    sennadatadir = "../data/senna/"

    tfidfdir = "../data/np/"
    
    datadir = "../../mead/data/C4_ShallowSummary_bigram/" 
    ShallowSummary(excelfile, datadir, sennadatadir, tfidfdir, np=None, method="bigram", K=4)
#     fio.newPath(tfidfdir)
#     
#     for np in ['chunk', 'syntax']:
#         getTFIDF(excelfile, tfidfdir, sennadatadir, np)
    
        #datadir = "../../mead/data/ShallowSummary_NPhraseSoft/" 
#     for np in ['chunk', 'syntax']:
#         datadir = "../../mead/data/ShallowSummary_NPhrase_"+np+"_TFIDF/"   
#         fio.deleteFolder(datadir)
#         ShallowSummary(excelfile, datadir, sennadatadir, tfidfdir, np, method="tfidf", K=30)
    
    for np in ['chunk', 'syntax']:
        datadir = "../../mead/data/Phrase_"+np+"_lexrank/"
        fio.deleteFolder(datadir)
        ShallowSummary(excelfile, datadir, sennadatadir, tfidfdir, np, method="lexrankmax", K=4)
    
#     for np in ['chunk', 'syntax']:
#         datadir = "../../mead/data/C4_Phrase_"+np+"_lexrank/"
#         fio.deleteFolder(datadir)
#         ShallowSummary(excelfile, datadir, sennadatadir, tfidfdir, np, method="lexrankmax", K=4)
        
    print "done"
    