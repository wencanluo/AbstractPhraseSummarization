import sys
import re
import fio
import xml.etree.ElementTree as ET
from collections import defaultdict
from Survey import *
import random
import NLTKWrapper
import tfidf
import phraseClusteringKmedoid

stopwordfilename = "../../../Fall2014/summarization/ROUGE-1.5.5/data/smart_common_words.txt"
stopwords = [line.lower().strip() for line in fio.readfile(stopwordfilename)]
print "stopwords:", len(stopwords)

stopwords = stopwords + ['.', '?', '-', ',', '[', ']', '-', ';', '\'', '"', '+', '&', '!', '/', '>', '<', ')', '(', '#', '=']

def getKeyNgram(student_summaryList, K=None, remove_stop = False, N = 5, weighted = False, M=1, save2file=None, ids=None):
    #K is the number of words to be extracted
    #N is the max number of Ngram
    key_ngrams = []
    
    Ndict = defaultdict(float)
    dict = defaultdict(float)
    for n in range(M, N+1):
        for summary in student_summaryList:
            ngrams = NLTKWrapper.getNgram(summary, n)
            for ngram in ngrams:
                ngram = ngram.lower()
                if n==1:
                    if ngram in stopwords: continue
                skip = False
                for stopword in stopwords:
                    if ngram.startswith(stopword + " "):
                        #print "skip:", ngram 
                        skip = True
                        break
                if skip: continue
                    
                if weighted:
                    dict[ngram] = dict[ngram] + n
                else:
                    dict[ngram] = dict[ngram] + 1
                Ndict[ngram] = n
    keys = sorted(dict, key=dict.get, reverse = True)
    
    if save2file != None:
        fio.SaveDict(dict, save2file, SortbyValueflag = True)
    
    return dict
            
def getKeyPhrases(student_summaryList, K, method='ngram', save2file=None, tfidfdir = None):
    if method == 'ngram':
        return getKeyNgram(student_summaryList, K, remove_stop = False, save2file=save2file)
    if method == 'unigram':
        return getKeyNgram(student_summaryList, K, remove_stop = False, N=1, save2file=save2file)
    if method == 'weightedngram':
        return getKeyNgram(student_summaryList, K, remove_stop = False, N=5, weighted = True, save2file=save2file)
    if method == "bigram":
        return getKeyNgram(student_summaryList, K, remove_stop = False, N=2, weighted = False, M=2, save2file=save2file)
    if method == 'ngram_remove_stop':
        return getKeyNgram(student_summaryList, K, remove_stop = True, save2file=save2file)
    if method == 'unigram_remove_stop':
        return getKeyNgram(student_summaryList, K, remove_stop = True, N=1, save2file=save2file)
    if method == 'weightedngram_remove_stop':
        return getKeyNgram(student_summaryList, K, remove_stop = True, N=5, weighted = True, save2file=save2file)
    if method == "bigram_remove_stop":
        return getKeyNgram(student_summaryList, K, remove_stop = True, N=2, weighted = False, M=2, save2file=save2file)
    return None
                            
def getShallowSummary(excelfile, folder, K=30, method='ngram', tfidfdir=None):
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
            
            Summary = []
            
            if method == 'unigram_tfidf':
                dict = fio.LoadDict(tfidfdir + str(week)+ '_' + type + ".keys")
            else:
                dict = getKeyPhrases(student_summaryList, K, method, save2file=filename + ".keys", tfidfdir=tfidfdir)
            
            keys = sorted(dict, key=dict.get, reverse = True)
            keys = phraseClusteringKmedoid.MalformedNPFlilter(keys)
            
            total_word = 0
            word_count = 0
            for key in keys:
                word_count = len(key.split())
                total_word = total_word + word_count
                #if total_word <= K:
                if len(Summary) + 1 <= K:
                    Summary.append(key)
            
            fio.savelist(Summary, filename)
            
def writeText(excelfile, folder):
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

            fio.newPath(folder)
            path = folder + str(week)+ '_'
            filename = path + type + '.txt'
            
            fio.savelist(student_summaryList, filename)

def computeTFIDF(folder):
    for type in ['POI', 'MP', 'LP']:
        my_tfidf = tfidf.TfIdf(stopword_filename=stopwordfilename)
        
        sheets = range(0,12)
        for i, sheet in enumerate(sheets):
            week = i + 1        
           
            path = folder + str(week)+ '_'
            filename = path + type + '.txt'
            
            wordstring = " ".join(NLTKWrapper.getWordList(filename))
            my_tfidf.add_input_document(wordstring)
        
        for i, sheet in enumerate(sheets):
            week = i + 1        
            path = folder + str(week)+ '_'
            filename = path + type + '.txt'
            
            wordstring = " ".join(NLTKWrapper.getWordList(filename))
            dict = my_tfidf.get_doc_keywords(wordstring)
            
            fio.SaveDict(dict, path + type + '.keys', SortbyValueflag = True)

def getTFIDF(excelfile, datadir):
    writeText(excelfile, datadir)
    computeTFIDF(datadir)
                        
def ShallowSummary(excelfile, datadir, K=30, method='ngram', tfidfdir=None):
    getShallowSummary(excelfile, datadir, K, method, tfidfdir)
    WriteTASummary(excelfile, datadir)
        
if __name__ == '__main__':
    excelfile = "../data/2011Spring.xls"
    
    sennadatadir = "../data/senna/"
    tfidfdir = "../data/tfidf/"
    #fio.deleteFolder(tfidfdir)
    getTFIDF(excelfile, tfidfdir)
    
    #getStudentResponses4Senna(excelfile, sennadatadir)
    
#     datadir = "../../mead/data/ShallowSummary_ngram/"  
#     fio.deleteFolder(datadir)
#     ShallowSummary(excelfile, datadir, K=30, method='ngram')
#     
#     datadir = "../../mead/data/ShallowSummary_unigram/"  
#     fio.deleteFolder(datadir)
#     ShallowSummary(excelfile, datadir, K=30, method='unigram')
#    
#     datadir = "../../mead/data/ShallowSummary_weightedngram/"  
#     fio.deleteFolder(datadir)
#     ShallowSummary(excelfile, datadir, K=30, method='weightedngram')
#     
#     datadir = "../../mead/data/ShallowSummary_bigram/"  
#     fio.deleteFolder(datadir)
#     ShallowSummary(excelfile, datadir, K=60, method='bigram')
    
#     datadir = "../../mead/data/ShallowSummary_ngram_remove_stop/"  
#     fio.deleteFolder(datadir)
#     ShallowSummary(excelfile, datadir, K=30, method='ngram_remove_stop')
#      
#     datadir = "../../mead/data/ShallowSummary_unigram_remove_stop/"  
#     fio.deleteFolder(datadir)
#     ShallowSummary(excelfile, datadir, K=30, method='unigram_remove_stop')
    
    for model in ['unigram_remove_stop', 'bigram']:
    #for model in ['unigram']:
        datadir = "../../mead/data/C4_"+model+"/"  
        fio.deleteFolder(datadir)
        ShallowSummary(excelfile, datadir, K=4, method=model, tfidfdir = tfidfdir)
#  
#     datadir = "../../mead/data/ShallowSummary_weightedngram_remove_stop/"  
#     fio.deleteFolder(datadir)
#     ShallowSummary(excelfile, datadir, K=30, method='weightedngram_remove_stop')
#      
#     datadir = "../../mead/data/ShallowSummary_bigram_remove_stop/"  
#     fio.deleteFolder(datadir)
#     ShallowSummary(excelfile, datadir, K=30, method='bigram_remove_stop')

    print 'done'
    