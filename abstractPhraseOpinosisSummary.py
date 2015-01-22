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

stopwords = [line.lower().strip() for line in fio.readfile(fio.stopwordfilename)]
print "stopwords:", len(stopwords)

stopwords = stopwords + ['.', '?', '-', ',', '[', ']', '-', ';', '\'', '"', '+', '&', '!', '/', '>', '<', ')', '(', '#', '=']
                    
def Write4Opinosis(student_summaryList, sennafile, output):
    sentences = SennaParser.SennaParse(sennafile)
    
    inputs = []
    for s in sentences:
        #wordpos = s.getWordPos(tolower=True)
        phrasewithPos = s.getPhrasewithPos()

        for p in phrasewithPos:
            if not p.endswith("./."):
                p = p + " ./."
            #p = "<s>/. " + p + " </s>/." #add a start and end point
            
            inputs.append(p)
        
    fio.SaveList(inputs, output, "\r\n")
                                        
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

            sennafile = sennadatadir + "senna." + str(week) + "." + type + '.output'
            
            filename = folder + str(week) + '_' + type + '.parsed'
            Write4Opinosis(student_summaryList, sennafile, filename)
                        
def ShallowSummary(excelfile, datadir, sennadatadir, tfidfdir, np, method, K=30):
    getShallowSummary(excelfile, datadir, sennadatadir, tfidfdir, np, method,  K)
    #WriteTASummary(excelfile, datadir)

def WriteSummary(excelfile, folder, datadirOpinosis):
    WriteTASummary(excelfile, folder)
    
    sheets = range(0,12)
    
    for i, sheet in enumerate(sheets):
        week = i + 1
        
        for type in ['POI', 'MP', 'LP']:
            filename = datadirOpinosis + "output/summary/" + str(week) + '_' + type + '.summary.system'
            
            Summary = fio.readfile(filename)
            
            Summary = [line.strip() for line in Summary]
            
            newS = []
            for s in Summary:
                if s.endswith(' .'):
                    s = s[:-2]
                newS.append(s) 
            Summary = newS
            
            path = folder + str(week)+ '/'
            fio.newPath(path)
            filename = path + type + '.summary'
            
            fio.SaveList(Summary, filename)
            
if __name__ == '__main__':
    excelfile = "../data/2011Spring.xls"
    
    sennadatadir = "../data/senna/"

    datadirOpinosis = "../../OpinosisSummarizer-1.0/phraseresponse/"
    #fio.newPath(datadirOpinosis)
    #ShallowSummary(excelfile, datadirOpinosis + "input/", sennadatadir, tfidfdir=None, np=None, method="Opinosis", K=4)
    
    datadir = "../../mead/data/OpinosisPhrase/"
    WriteSummary(excelfile, datadir, datadirOpinosis)
          
    print "done"
    