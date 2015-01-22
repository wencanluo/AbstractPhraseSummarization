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
import postProcess

stopwords = [line.lower().strip() for line in fio.readfile(W)]
print "stopwords:", len(stopwords)

stopwords = stopwords + ['.', '?', '-', ',', '[', ']', '-', ';', '\'', '"', '+', '&', '!', '/', '>', '<', ')', '(', '#', '=']
                    
def Write4Opinosis(student_summaryList, sennafile, clusterfile, week, sources, output):
    #get NP, POS dictionary
    
    dict = {}
    sentences = SennaParser.SennaParse(sennafile)
    for s in sentences:
        phrases = s.getPhrasPairwithPos()

        for np, p in phrases:
            if not p.endswith("./."):
                p = p + " ./." #add a start and end point
            dict[np.lower()] = p
    
    body = fio.readMatrix(clusterfile, False)
            
    NPs = [row[0] for row in body]
    clusterids = [row[1] for row in body]
    
    #rank by number of students
    keys = postProcess.RankCluster2(NPs, None, clusterids, sources)
    
    M = 0
    for key in keys:
        #get the NPs
        
        cluster = [i for i in range(len(clusterids)) if clusterids[i] == key]
        
        inputs = []
        for i in cluster:
            NP = NPs[i]
            if NP not in dict:
                print NP, "error"
            else:
                inputs.append(dict[NP])
        
        fio.SaveList(inputs, output + "_" + str(M) + ".parsed", "\r\n")
        M = M + 1
                                        
def getShallowSummary(excelfile, folder, sennadatadir, clusterdir, tfidfdir, np, method, K=30, ratio='sqrt'):
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
            student_summaryList = getStudentResponseList(orig, header, summarykey, type, withSource=True)
            ids = [summary[1] for summary in student_summaryList]
            summaries = [summary[0] for summary in student_summaryList]
            
            sennafile = sennadatadir + "senna." + str(week) + "." + type + '.output'
            output = clusterdir + str(week) +'/' + type + ".cluster.kmedoids." + str(ratio) + "." +method + '.' + np
            weightfile = clusterdir + str(week)+ '/' + type + '.' + np + '.' + method
            
            if not fio.isExist(output):
                phraseClusteringKmedoid.getPhraseClusterAll(sennafile, weightfile, output, ratio, MalformedFlilter=True, source=ids, np=np)
                
            filename = folder + str(week) + '_' + type
            Write4Opinosis(student_summaryList, sennafile, output, week, ids, filename)
                        
def ShallowSummary(excelfile, datadir, sennadatadir, clusterdir, tfidfdir, np, method, K=30):
    getShallowSummary(excelfile, datadir, sennadatadir, clusterdir, tfidfdir, np, method,  K)
    #WriteTASummary(excelfile, datadir)

def WriteSummary(excelfile, folder, datadirOpinosis):
    WriteTASummary(excelfile, folder)
    
    sheets = range(0,12)
    
    for i, sheet in enumerate(sheets):
        week = i + 1
        
        for type in ['POI', 'MP', 'LP']:
            Summary = []
            
            for k in range(4):
                filename = datadirOpinosis + "output/summary/" + str(week) + '_' + type + "_"+ str(k) + '.summary.system'
            
                lines = fio.readfile(filename)
                
                if len(lines) == 0: 
                    print filename
                    continue
                
                Summary.append(lines[0].strip())
            
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

    datadirOpinosis = "../../OpinosisSummarizer-1.0/phrasecluster/"
    fio.newPath(datadirOpinosis)
    
    i=511
    clusterdir = "../data/np"+str(i)+"/"
    
    #ShallowSummary(excelfile, datadirOpinosis + "input/", sennadatadir, clusterdir, tfidfdir=None, np='syntax', method='optimumComparerLSATasa', K=4)
    
    datadir = "../../mead/data/OpinosisPhraseCluster/"
    WriteSummary(excelfile, datadir, datadirOpinosis)
          
    print "done"
    