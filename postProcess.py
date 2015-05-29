from OrigReader import prData
import sys
import re
import fio
import xml.etree.ElementTree as ET
import numpy
from collections import defaultdict
import NLTKWrapper
import phraseClusteringKmedoid
import json
import shallowSummary

from Survey import *
                                        
def formatSummaryOutput(excelfile, datadir, output):
    reload(sys)
    sys.setdefaultencoding('utf8')
    
    header = ['ID', 'Gender', 'Point of Interest', 'Muddiest Point', 'Learning Point']
    summarykey = "Top Answers"
    
    sheets = range(0,12)
    
    datahead = ['Week', '# of sentence POI', '# of sentence POI', '# of sentence POI']
    
    head = ['Week', 'TA:POI', 'S:POI','TA:MP','S:MP','TA:LP', 'S:LP',]
    body = []
    
    for sheet in sheets:
        row = []
        week = sheet + 1
        row.append(week)
            
        for type in ['POI', 'MP', 'LP']:
            summaries = getMeadSummary(datadir, type)
            
            orig = prData(excelfile, sheet)
        
            summary = getTASummary(orig, header, summarykey, type)
            if summary == []:
                row.append("")
            else:
                row.append(";".join(summary))
                
            summary = ";".join(summaries[sheet])
            row.append(summary)
        
        body.append(row)
            
    fio.WriteMatrix(output, body, head)
    
def GetRougeScore(datadir, models, outputdir):
    for model in models:
        print model
        
        #sheets = range(0,12)
        sheets = [1,2,3,4,5,6,7,10]
        types = ['POI', 'MP', 'LP']
        scores = ['ROUGE-1','ROUGE-2', 'ROUGE-SUX']
        
        #header = ['week', 'R1', 'R2', 'R-SU4', 'R1', 'R2', 'R-SU4', 'R1', 'R2', 'R-SU4']
        #header = ['week', 'R1-R', 'R1-P', 'R1-F', 'R2-R', 'R2-P', 'R2-F', 'RSU4-R', 'RSU4-P', 'RSU4-F', 'R1-R', 'R1-P', 'R1-F', 'R2-R', 'R2-P', 'R2-F', 'RSU4-R', 'RSU4-P', 'RSU4-F','R1-R', 'R1-P', 'R1-F', 'R2-R', 'R2-P', 'R2-F', 'RSU4-R', 'RSU4-P', 'RSU4-F',]
        header = ['week', 'R1-R', 'R1-P', 'R1-F', 'R2-R', 'R2-P', 'R2-F', 'RSU4-R', 'RSU4-P', 'RSU4-F',]
        
        body = []
        for sheet in sheets:
            for type in types:
                week = sheet + 1
                path = datadir + model + '/' + str(week)+ '/'
                fio.NewPath(path)
        
                row = []
                row.append(week)
                for scorename in scores:
                    filename = path + type + "_OUT_"+scorename+".csv"
                    lines = fio.ReadFile(filename)
                    try:
                        scorevalues = lines[1].split(',')
                        score = scorevalues[1].strip()
                        row.append(score)
                        score = scorevalues[2].strip()
                        row.append(score)
                        score = scorevalues[3].strip()
                        row.append(score)
                    except Exception:
                        print filename, scorename, lines
                body.append(row)
        
        #get average
        
        row = []
        row.append("average")
        for i in range(1, len(header)):
            scores = [float(xx[i]) for xx in body]
            row.append(numpy.mean(scores))
        body.append(row)
        
        fio.WriteMatrix(outputdir + "rouge." + model + ".txt", body, header)
        
def GetRougeScoreSingle(datadir, models, outputdir):
    for model in models:
        print model
        sheets = range(0,12)
        types = ['POI', 'MP', 'LP']
        scores = ['ROUGE-1','ROUGE-2', 'ROUGE-SUX']
        
        #header = ['week', 'R1', 'R2', 'R-SU4', 'R1', 'R2', 'R-SU4', 'R1', 'R2', 'R-SU4']
        header = ['week', 'R1-R', 'R1-P', 'R1-F', 'R2-R', 'R2-P', 'R2-F', 'RSU4-R', 'RSU4-P', 'RSU4-F', 'R1-R', 'R1-P', 'R1-F', 'R2-R', 'R2-P', 'R2-F', 'RSU4-R', 'RSU4-P', 'RSU4-F','R1-R', 'R1-P', 'R1-F', 'R2-R', 'R2-P', 'R2-F', 'RSU4-R', 'RSU4-P', 'RSU4-F',]
        
        body = []
        for sheet in sheets:
            week = sheet + 1
            path = datadir + model + '/' + str(week)+ '/'
            fio.NewPath(path)
            
            row = []
            row.append(week)
            for type in types:
                for scorename in scores:
                    filename = path + type + "_OUT_"+scorename+".csv"
                    lines = fio.ReadFile(filename)
                    try:
                        scorevalues = lines[1].split(',')
                        score = scorevalues[1].strip()
                        row.append(score)
                        score = scorevalues[2].strip()
                        row.append(score)
                        score = scorevalues[3].strip()
                        row.append(score)
                    except Exception:
                        print filename, scorename, lines
            body.append(row)
        
        #get average
        
        row = []
        row.append("average")
        for i in range(1, len(header)):
            scores = [float(xx[i]) for xx in body]
            row.append(numpy.mean(scores))
        body.append(row)
        
        fio.WriteMatrix(outputdir + "rouge." + model + ".txt", body, header)

def GetRougeScoreMMRSingle(datadir, models, outputdir): #only keep the average
    for model in models:
        sheets = range(0,12)
        types = ['POI', 'MP', 'LP']
        scores = ['ROUGE-1','ROUGE-2', 'ROUGE-SUX']
        
        header = ['week', 'R1-P', 'R1-R', 'R1-F', 'R2-P', 'R2-R', 'R2-F', 'RSU4-P', 'RSU4-R', 'RSU4-F', 'R1-P', 'R1-R', 'R1-F', 'R2-P', 'R2-R', 'R2-F', 'RSU4-P', 'RSU4-R', 'RSU4-F', 'R1-P', 'R1-R', 'R1-F', 'R2-P', 'R2-R', 'R2-F', 'RSU4-P', 'RSU4-R', 'RSU4-F', ]
        #header = ['lamda', 'R1', 'R2', 'R-SU4', 'R1', 'R2', 'R-SU4', 'R1', 'R2', 'R-SU4']
        averagebody = []
        
        for r in ['0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0']:
            body = []
            for sheet in sheets:
                week = sheet + 1
                path = datadir + model + '/' + str(week)+ '/'
                fio.NewPath(path)
                
                row = []
                row.append(week)
                for type in types:
                    for scorename in scores:
                        filename = path + type + "." + str(r) + "_OUT_"+scorename+".csv"
                        lines = fio.ReadFile(filename)
                        try:
                            scorevalues = lines[1].split(',')
                            score = scorevalues[1].strip()
                            row.append(score)
                            score = scorevalues[2].strip()
                            row.append(score)
                            score = scorevalues[3].strip()
                            row.append(score)
                        except Exception:
                            print filename, scorename, lines
                body.append(row)
            
            #get average
            
            row = []
            row.append("average")
            arow = []
            arow.append('mmr_lambda_'+r)
            for i in range(1, len(header)):
                ave = [float(xx[i]) for xx in body]
                row.append(numpy.mean(ave))
                arow.append(numpy.mean(ave))
            body.append(row)
            averagebody.append(arow)
  
            fio.WriteMatrix(outputdir + "rouge." + model + '.' + r + ".txt", body, header)
        
        #get max
        #get the max
        row = []
        row.append("max")
        for i in range(1, len(header)):
            scores = [float(xx[i]) for xx in averagebody]
            row.append(numpy.max(scores))
        averagebody.append(row)
        
        fio.WriteMatrix(outputdir + "rouge." + model + ".txt", averagebody, header)
                
def GetRougeScoreMMR(datadir, models, outputdir): #only keep the average
    for model in models:
        sheets = range(0,12)
        types = ['POI', 'MP', 'LP']
        scores = ['ROUGE-1','ROUGE-2', 'ROUGE-SUX']
        
        header = ['week', 'R1-R', 'R1-P', 'R1-F', 'R2-R', 'R2-P', 'R2-F', 'RSU4-R', 'RSU4-P', 'RSU4-F']
        #header = ['week', 'R1-P', 'R1-R', 'R1-F', 'R2-P', 'R2-R', 'R2-F', 'RSU4-P', 'RSU4-R', 'RSU4-F', 'R1-P', 'R1-R', 'R1-F', 'R2-P', 'R2-R', 'R2-F', 'RSU4-P', 'RSU4-R', 'RSU4-F', 'R1-P', 'R1-R', 'R1-F', 'R2-P', 'R2-R', 'R2-F', 'RSU4-P', 'RSU4-R', 'RSU4-F', ]
        #header = ['lamda', 'R1', 'R2', 'R-SU4', 'R1', 'R2', 'R-SU4', 'R1', 'R2', 'R-SU4']
        averagebody = []
        
        for r in ['0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0']:
            body = []
            for type in types:
                for sheet in sheets:
                    week = sheet + 1
                    path = datadir + model + '/' + str(week)+ '/'
                    fio.NewPath(path)
                
                    row = []
                    row.append(week)
                    for scorename in scores:
                        filename = path + type + "." + str(r) + "_OUT_"+scorename+".csv"
                        lines = fio.ReadFile(filename)
                        try:
                            scorevalues = lines[1].split(',')
                            score = scorevalues[1].strip()
                            row.append(score)
                            score = scorevalues[2].strip()
                            row.append(score)
                            score = scorevalues[3].strip()
                            row.append(score)
                        except Exception:
                            print filename, scorename, lines
                    body.append(row)
            
            #get average
            
            row = []
            row.append("average")
            arow = []
            arow.append(model + '_'+r)
            for i in range(1, len(header)):
                ave = [float(xx[i]) for xx in body]
                row.append(numpy.mean(ave))
                arow.append(numpy.mean(ave))
            body.append(row)
            averagebody.append(arow)
  
            fio.WriteMatrix(outputdir + "rouge." + model + '.' + r + ".txt", body, header)
        
        #get max
        #get the max
        row = []
        row.append("max")
        for i in range(1, len(header)):
            scores = [float(xx[i]) for xx in averagebody]
            row.append(numpy.max(scores))
        averagebody.append(row)
        
        fio.WriteMatrix(outputdir + "rouge." + model + ".txt", averagebody, header)
           
def getWordCount(summary, output):
    head, body = fio.ReadMatrix(summary, True)
    
    data = []
    
    for row in body:
        newrow = []
        for i in range(len(head)):
            if i==0: continue
            newrow.append( len(row[i].split()) )
        
        data.append(newrow)
    
    newhead = []
    for i in range(len(head)):
        if i==0: continue
        newhead.append("WC_"+head[i])
    
    fio.WriteMatrix(output, data, newhead)
    
def getTAWordCountDistribution(excelfile, output):
    reload(sys)
    sys.setdefaultencoding('utf8')
    
    header = ['ID', 'Gender', 'Point of Interest', 'Muddiest Point', 'Learning Point']
    summarykey = "Top Answers"
    
    sheets = range(0,12)
    
    datahead = ['Week', '# of sentence POI', '# of sentence POI', '# of sentence POI']
    
    counts = []
    
    for sheet in sheets:
        row = []
        week = sheet + 1
        
        orig = prData(excelfile, sheet)
        
        for type in ['POI', 'MP', 'LP']:
            summaryList = getTASummary(orig, header, summarykey, type)
            
            N = 0
            for summary in summaryList:
                N = N + len(summary.split())
                
            if N>100:
                print type, week, summaryList
                
            counts.append(N)
    
    
    print numpy.max(counts),'\t',numpy.min(counts),'\t',numpy.mean(counts),'\t',numpy.median(counts),'\t',numpy.std(counts)
 
def getTALengthDistribution(excelfile, output):
    reload(sys)
    sys.setdefaultencoding('utf8')
    
    header = ['ID', 'Gender', 'Point of Interest', 'Muddiest Point', 'Learning Point']
    summarykey = "Top Answers"
    
    sheets = range(0,12)
    
    datahead = ['Week', '# of sentence POI', '# of sentence POI', '# of sentence POI']
    
    counts = []
    
    for sheet in sheets:
        row = []
        week = sheet + 1
        
        orig = prData(excelfile, sheet)
        
        for type in ['POI', 'MP', 'LP']:
            summaryList = getTASummary(orig, header, summarykey, type)
            
            counts.append(len(summaryList))
    
    fio.PrintList(counts, ", ")
    print numpy.max(counts),'\t',numpy.min(counts),'\t',numpy.mean(counts),'\t',numpy.median(counts),'\t',numpy.std(counts)
 
    
def getTAWordCountDistribution2(excelfile, output):
    reload(sys)
    sys.setdefaultencoding('utf8')
    
    header = ['ID', 'Gender', 'Point of Interest', 'Muddiest Point', 'Learning Point']
    summarykey = "Top Answers"
    
    sheets = range(0,12)
    
    datahead = ['Week', '# of sentence POI', '# of sentence POI', '# of sentence POI']
    
    counts = {'POI':{}, 'MP':{}, 'LP':{}}
    for sheet in sheets:
        row = []
        week = sheet + 1
        
        orig = prData(excelfile, sheet)
        
        for type in ['POI', 'MP', 'LP']:
            summaryList = getTASummary(orig, header, summarykey, type)
            for summary in summaryList:
                counts[type][summary]  = len(summary.split())
    
    for type in ['POI', 'MP', 'LP']:
        print numpy.max(counts[type].values()),'\t',numpy.min(counts[type].values()),'\t',numpy.mean(counts[type].values()),'\t',numpy.median(counts[type].values()),'\t',numpy.std(counts[type].values())
        #fio.PrintList(counts[type].values(), sep='\t')
        #fio.PrintDict(counts[type], True)
        #print
        
def getMeadAverageWordCount(summary, output):
    counts = {'POI':{}, 'MP':{}, 'LP':{}}
    
    for type in ['POI', 'MP', 'LP']:
        summaries = getMeadSummary(datadir, type)
        for weeksummary in summaries:
            for summary in weeksummary:
                counts[type][summary]=len(summary.split())
    
    fio.PrintList(["Type", "Max", "Min", "Mean", "Median", "Std"], "\t")
    for type in ['POI', 'MP', 'LP']:
        #fio.PrintList(counts[type].values(), sep=',')
        print type,'\t',numpy.max(counts[type].values()),'\t',numpy.min(counts[type].values()),'\t',numpy.mean(counts[type].values()),'\t',numpy.median(counts[type].values()),'\t',numpy.std(counts[type].values())

def getStudentResponseAverageWords(excelfile, output):
    header = ['ID', 'Gender', 'Point of Interest', 'Muddiest Point', 'Learning Point']
    summarykey = "Top Answers"
    
    #sheets = range(0,25)
    sheets = range(0,12)
    
    body = []
    for i, sheet in enumerate(sheets):
        week = i + 1
        orig = prData(excelfile, sheet)
        
        counts = {'POI':{}, 'MP':{}, 'LP':{}}
        
        row = []
        row.append(week)
        
        for type in ['POI', 'MP', 'LP']:
            summaries = getStudentResponse(orig, header, summarykey, type=type)
            for summaryList in summaries.values():
                for s in summaryList:
                    counts[type][s] = len(s.split())
        
            row.append(numpy.mean(counts[type].values()))
            row.append(numpy.std(counts[type].values()))
        body.append(row)
            
    fio.WriteMatrix(output, body, ['Week', 'POI', '', 'MP', '', 'LP', '']) 

def getStudentResponseWordCountDistribution2(excelfile, output):
    header = ['ID', 'Gender', 'Point of Interest', 'Muddiest Point', 'Learning Point']
    summarykey = "Top Answers"
    
    sheets = range(0,25)
    #sheets = range(0,12)
    
    counts = {'POI':{}, 'MP':{}, 'LP':{}}
    
    dict = {'POI':{}, 'MP':{}, 'LP':{}}
    
    AveBody = []
    AveHead = ['week', 'POI', 'MP', 'LP']
    
    body = []
    for i, sheet in enumerate(sheets):
        week = i + 1
        orig = prData(excelfile, sheet)
        
        AveRow = [week]
        
        row = []
        for type in ['POI', 'MP', 'LP']:
            summaries = getStudentResponse(orig, header, summarykey, type=type)
            
            totalN = 0
            totalWordCount = 0
        
            for summaryList in summaries.values():
                N = 0
                for s in summaryList:
                    nlen = len(NLTKWrapper.wordtokenizer(s, punct=True))
                    counts[type][s] = nlen
                    N = N + nlen
                
                if N == 0: continue
                
                if N not in dict[type]:
                    dict[type][N] = 0
                dict[type][N] = dict[type][N] + 1
                
                totalWordCount = totalWordCount + N
                totalN = totalN + 1
                
            print totalWordCount, totalN
            AveRow.append(float(totalWordCount)/totalN)
        
        AveBody.append(AveRow)
    
    #fio.PrintDict(dict['MP'], SortbyValueflag=False)
    
    fio.WriteMatrix("../data/wordcount.txt", AveBody, AveHead)
    
    values = []
    for key in range(0, 46):
        if key in dict['MP']:
            values.append(dict['MP'][key])
        else:
            values.append(0)
    fio.PrintList(values, ',')
        
    #for type in ['POI', 'MP', 'LP']:
    for type in ['MP']:
        print numpy.max(counts[type].values()),'\t',numpy.min(counts[type].values()),'\t',numpy.mean(counts[type].values()),'\t',numpy.median(counts[type].values()),'\t',numpy.std(counts[type].values())
        fio.PrintList(counts[type].values(), sep=',')
        #fio.PrintDict(counts[type], True)
        #print
        
def getStudentResponseWordCountDistribution(excelfile, output):
    header = ['ID', 'Gender', 'Point of Interest', 'Muddiest Point', 'Learning Point']
    summarykey = "Top Answers"
    
    #sheets = range(0,25)
    sheets = range(0,12)
    
    counts = []
    
    dict = {}
    
    body = []
    for i, sheet in enumerate(sheets):
        week = i + 1
        orig = prData(excelfile, sheet)
        
        row = []
        for type in ['POI', 'MP', 'LP']:
            summaries = getStudentResponse(orig, header, summarykey, type=type)
            
            for summaryList in summaries.values():
                N = 0
                for s in summaryList:   
                    N = N + len(s.split())
                
                if N==0: continue
                counts.append(N)
  
                if N not in dict:
                    dict[N] = 0
                dict[N] = dict[N] + 1
    
    #fio.PrintDict(dict['MP'], SortbyValueflag=False)
    
    values = []
    for key in range(0, 55):
        if key in dict:
            values.append(dict[key])
        else:
            values.append(0)
    fio.PrintList(values, ',')
        
    #for type in ['POI', 'MP', 'LP']:
    print numpy.max(counts),'\t',numpy.min(counts),'\t',numpy.mean(counts),'\t',numpy.median(counts),'\t',numpy.std(counts)
    #fio.PrintList(counts.values(), sep=',')
        #fio.PrintDict(counts[type], True)
        #print

def CheckKeyword(keyword, sentences):
    for s in sentences:
        if s.lower().find(keyword.lower()) != -1:
            return True
    return False

def TASummaryCoverage2(excelfile, output):
    reload(sys)
    sys.setdefaultencoding('utf8')
    
    header = ['ID', 'Gender', 'Point of Interest', 'Muddiest Point', 'Learning Point']
    summarykey = "Top Answers"
    
    sheets = range(0,12)
    
    datahead = ['Week', '# of sentence POI', '# of sentence POI', '# of sentence POI']
    
    head = ['Week', 'TA:POI', 'S:POI','TA:MP','S:MP','TA:LP', 'S:LP',]
    body = []
    
    MaxNgram = 5
    dict = {'POI':{}, 'MP':{}, 'LP':{}}
    
    uncoveried = defaultdict(float)
    
    for type in ['POI', 'MP', 'LP']:
        for n in range(MaxNgram):
            dict[type][n+1] = [0,0]
    
    for sheet in sheets:
        row = []
        week = sheet + 1
        row.append(week)
        
        for type in ['POI', 'MP', 'LP']:
            orig = prData(excelfile, sheet)
        
            student_summaries = getStudentResponse(orig, header, summarykey, type)
            student_summaryList = []
            
            for summaryList in student_summaries.values():
                for s in summaryList:
                    student_summaryList.append(s)
            
            ta_summaries = getTASummary(orig, header, summarykey, type)
            
            for summary in ta_summaries:
                for n in range(MaxNgram):
                    ngrams = NLTKWrapper.getNgram(summary, n+1)
                    dict[type][n+1][0] = dict[type][n+1][0] + len(ngrams)
                    
                    for token in ngrams:
                        if CheckKeyword(token, student_summaryList):
                            dict[type][n+1][1] = dict[type][n+1][1] + 1
                        else:
                            uncoveried[token.lower()] = uncoveried[token.lower()] + 1
        
    #fio.PrintList(["Question", "Ngram Length", "# of points", "# of response points", "coverage ratio"])
    fio.PrintList(["Question", "Ngram Length", "coverage ratio"])
    for type in ['POI', 'MP', 'LP']:
        for n in range(MaxNgram):
            #print type, "\t", n+1, "\t", dict[type][n+1][0], "\t", dict[type][n+1][1], "\t", float(dict[type][n+1][1])/dict[type][n+1][0]
            r = float(dict[type][n+1][1])/float(dict[type][n+1][0])
            #print type, "\t", n+1, "\t", "%.2f" % r
            print "%.2f," % r, 
        print
    
    fio.PrintDict(uncoveried, True)
        
def TASummaryCoverage(excelfile, output):
    reload(sys)
    sys.setdefaultencoding('utf8')
    
    header = ['ID', 'Gender', 'Point of Interest', 'Muddiest Point', 'Learning Point']
    summarykey = "Top Answers"
    
    sheets = range(0,12)
    
    datahead = ['Week', '# of sentence POI', '# of sentence POI', '# of sentence POI']
    
    head = ['Week', 'TA:POI', 'S:POI','TA:MP','S:MP','TA:LP', 'S:LP',]
    body = []
    
    MaxNgram = 5
    dict = {}
    
    uncoveried = defaultdict(float)
    
    for type in ['POI', 'MP', 'LP']:
        for n in range(MaxNgram):
            dict[n+1] = [0,0]
    
    for sheet in sheets:
        row = []
        week = sheet + 1
        row.append(week)
        
        for type in ['POI', 'MP', 'LP']:
            orig = prData(excelfile, sheet)
        
            student_summaries = getStudentResponse(orig, header, summarykey, type)
            student_summaryList = []
            
            for summaryList in student_summaries.values():
                for s in summaryList:
                    student_summaryList.append(s)
            
            ta_summaries = getTASummary(orig, header, summarykey, type)
            
            for summary in ta_summaries:
                for n in range(MaxNgram):
                    ngrams = NLTKWrapper.getNgram(summary, n+1)
                    dict[n+1][0] = dict[n+1][0] + len(ngrams)
                    
                    for token in ngrams:
                        if CheckKeyword(token, student_summaryList):
                            dict[n+1][1] = dict[n+1][1] + 1
                        else:
                            uncoveried[token.lower()] = uncoveried[token.lower()] + 1
        
    #fio.PrintList(["Question", "Ngram Length", "# of points", "# of response points", "coverage ratio"])
    fio.PrintList(["Question", "Ngram Length", "coverage ratio"])
    #for type in ['POI', 'MP', 'LP']:
    for n in range(MaxNgram):
        #print type, "\t", n+1, "\t", dict[n+1][0], "\t", dict[n+1][1], "\t", float(dict[n+1][1])/dict[n+1][0]
        r = float(dict[n+1][1])/float(dict[n+1][0])
        #print type, "\t", n+1, "\t", "%.2f" % r
        print "%.2f," % r, 
    
    fio.PrintDict(uncoveried, True)

def ExtractNP(datadir, outdir, method="syntax"):
    sheets = range(0,12)
    
    for i, sheet in enumerate(sheets):
        week = i + 1
        
        for type in ['POI', 'MP', 'LP']:
            
            file = datadir + str(week)+ '/' + type + '.summary.keys'
            if not fio.isExist(file): continue
            
            dict = fio.LoadDict(file, 'float')
            keys = sorted(dict, key=dict.get, reverse = True)
            
            fio.NewPath(outdir + str(week)+ '/')
            output = outdir + str(week)+ '/' + type + '.'+method+'.key'
            fio.SaveList(keys, output)

def ExtractNPFromRaw(excelfile, sennadatadir, outdir, method="syntax", weekrange=range(0,12), Split=True):
    sheets = weekrange
    
    header = ['ID', 'Gender', 'Point of Interest', 'Muddiest Point', 'Learning Point']
    summarykey = "Top Answers"
    
    for i, sheet in enumerate(sheets):
        week = i + 1
        
        orig = prData(excelfile, sheet)
        
        for type in ['POI', 'MP', 'LP']:
            
            sennafile = sennadatadir + "senna." + str(week) + "." + type + '.output'
            
            student_summaryList = getStudentResponseList(orig, header, summarykey, type, withSource=True, Split=Split)
            ids = [summary[1] for summary in student_summaryList]
            NPs, sources = phraseClusteringKmedoid.getNPs(sennafile, MalformedFlilter=False, source=ids, np=method)
            
            keys = set(NPs)
            
            fio.NewPath(outdir + str(week)+ '/')
            output = outdir + str(week)+ '/' + type + '.'+method+'.key'
            fio.SaveList(keys, output)

import Survey

def ExtractQualityScore(excelfile, sennadatadir, outdir, method="syntax", weekrange=range(0,12), Split=True):
    sheets = weekrange
    
    header = ['ID', 'Gender', 'Point of Interest', 'Muddiest Point', 'Learning Point']
    summarykey = "Top Answers"
    
    for i, sheet in enumerate(sheets):
        week = i + 1
        
        orig = prData(excelfile, sheet)
        
        for type in ['MP']:
            
            sennafile = sennadatadir + "senna." + str(week) + "." + type + '.output'
            
            student_summaryList = getStudentResponseList(orig, header, summarykey, type, withSource=True, Split=Split)
            ids = [summary[1] for summary in student_summaryList]
            NPs, sources = phraseClusteringKmedoid.getNPs(sennafile, MalformedFlilter=False, source=ids, np=method)
            
            qality_dict = Survey.getStudentQualityDict(orig, header, summarykey)
            
            dict = {}
            for np, source in zip(NPs, sources):
                if source in qality_dict:
                    dict[np] = qality_dict[source]

            fio.NewPath(outdir + str(week)+ '/')
            output = outdir + str(week)+ '/' + type + '.'+method+'.quality'        
            fio.SaveDict(dict, output)
            
def ExtractNPFromRawWithCount(excelfile, sennadatadir, outdir, method="syntax", weekrange=range(0,12), Split=True):
    sheets = weekrange
    
    header = ['ID', 'Gender', 'Point of Interest', 'Muddiest Point', 'Learning Point']
    summarykey = "Top Answers"
    
    for i, sheet in enumerate(sheets):
        week = i + 1
        
        orig = prData(excelfile, sheet)
        
        for type in ['POI', 'MP', 'LP']:
            
            sennafile = sennadatadir + "senna." + str(week) + "." + type + '.output'
            
            student_summaryList = getStudentResponseList(orig, header, summarykey, type, withSource=True, Split=Split)
            ids = [summary[1] for summary in student_summaryList]
            NPs, sources = phraseClusteringKmedoid.getNPs(sennafile, MalformedFlilter=False, source=ids, np=method)
            
            dict = {}
            
            for NP in NPs:
                if NP not in dict:
                    dict[NP] = 0
                dict[NP] = dict[NP] + 1
                        
            fio.NewPath(outdir + str(week)+ '/')
            output = outdir + str(week)+ '/' + type + '.'+method+'.dict'
            fio.SaveDict(dict, output)
            
def ExtractNPSource(excelfile, sennadatadir, outdir, method="syntax", weekrange=range(0,12), Split=True):
    sheets = weekrange
    
    header = ['ID', 'Gender', 'Point of Interest', 'Muddiest Point', 'Learning Point']
    summarykey = "Top Answers"
    
    for i, sheet in enumerate(sheets):
        week = i + 1
        
        orig = prData(excelfile, sheet)
        
        for type in ['POI', 'MP', 'LP']:
            
            sennafile = sennadatadir + "senna." + str(week) + "." + type + '.output'
            
            student_summaryList = getStudentResponseList(orig, header, summarykey, type, withSource=True, Split=Split)
            ids = [summary[1] for summary in student_summaryList]
            NPs, sources = phraseClusteringKmedoid.getNPs(sennafile, MalformedFlilter=False, source=ids, np=method)
            
            dict = {}
            for np, id in zip(NPs, sources):
                if np not in dict:
                    dict[np] = []
                dict[np].append(id)
            
            fileout = outdir + str(week)+ '/' + type + '.'+method+'.keys.source'
            
            with open(fileout, 'w') as outfile:
                json.dump(dict, outfile, indent=2)

def ExtractUnigramSource(excelfile, outdir, method="unigram"):
    sheets = range(0,12)
    
    header = ['ID', 'Gender', 'Point of Interest', 'Muddiest Point', 'Learning Point']
    summarykey = "Top Answers"
    
    for i, sheet in enumerate(sheets):
        week = i + 1
        
        orig = prData(excelfile, sheet)
        
        for type in ['POI', 'MP', 'LP']:
            
            sennafile = sennadatadir + "senna." + str(week) + "." + type + '.output'
            
            student_summaryList = getStudentResponseList(orig, header, summarykey, type, withSource=True)
            
            dict = {}
            for summary, id in student_summaryList:
                ngrams = NLTKWrapper.getNgram(summary, 1)
                for ngram in ngrams:
                    ngram = ngram.lower()
                    if ngram not in dict:
                        dict[ngram] = []
                    dict[ngram].append(id)
            
            fileout = outdir + str(week)+ '/' + type + '.'+method+'.keys.source'
            
            with open(fileout, 'w') as outfile:
                json.dump(dict, outfile)
               
def CombineKMethod(datadir, output, methods, ratios, nps, model_prefix):
    Header = ['np', 'method', 'lambda', 'R1-R', 'R1-P', 'R1-F', 'R2-R', 'R2-P', 'R2-F', 'RSU4-R', 'RSU4-P', 'RSU4-F', 'R1-R', 'R1-P', 'R1-F', 'R2-R', 'R2-P', 'R2-F', 'RSU4-R', 'RSU4-P', 'RSU4-F', 'R1-R', 'R1-P', 'R1-F', 'R2-R', 'R2-P', 'R2-F', 'RSU4-R', 'RSU4-P', 'RSU4-F' ]
    newbody = []
    
    for np in nps:
        for method in methods: 
            for ratio in ratios:
                filename = datadir + "rouge." + model_prefix + "_" + str(ratio) + "_"+ method + '_' + np + ".txt"
                head, body = fio.ReadMatrix(filename, hasHead=True)
                
                row = []
                row.append(np)
                row.append(method)
                row.append(ratio)
                row = row + body[-1][1:]
                
                newbody.append(row)
            
    #get the max
    row = []
    row.append("max")
    row.append("")
    row.append("")
    for i in range(3, len(Header)):
        scores = [float(xx[i]) for xx in newbody]
        row.append(numpy.max(scores))
    newbody.append(row)
    
    fio.WriteMatrix(output, newbody, Header)

def CombineRouges(models, outputdir):
    Header = ['method', 'R1-R', 'R1-P', 'R1-F', 'R2-R', 'R2-P', 'R2-F', 'RSU4-R', 'RSU4-P', 'RSU4-F',]
    newbody = []
    
    for model in models: 
        filename = outputdir + "rouge." + model + ".txt"
        head, body = fio.ReadMatrix(filename, hasHead=True)
        
        row = []
        row.append(model)
        row = row + body[-1][1:]
        
        newbody.append(row)
            
    #get the max
    row = []
    row.append("max")
    for i in range(1, len(Header)):
        scores = [float(xx[i]) for xx in newbody]
        row.append(numpy.max(scores))
    newbody.append(row)
    
    newname = outputdir + "_".join(models) + ".txt"
    if len(newname) > 50:
        newname = newname[:50] + "_50.txt"
    fio.WriteMatrix(newname, newbody, Header) 
        
def CombineRouges2(models, outputdir):
    Header = ['method', 'R1-R', 'R1-P', 'R1-F', 'R2-R', 'R2-P', 'R2-F', 'RSU4-R', 'RSU4-P', 'RSU4-F', 'R1-R', 'R1-P', 'R1-F', 'R2-R', 'R2-P', 'R2-F', 'RSU4-R', 'RSU4-P', 'RSU4-F', 'R1-R', 'R1-P', 'R1-F', 'R2-R', 'R2-P', 'R2-F', 'RSU4-R', 'RSU4-P', 'RSU4-F']
    newbody = []
    
    for model in models: 
        filename = outputdir + "rouge." + model + ".txt"
        head, body = fio.ReadMatrix(filename, hasHead=True)
        
        row = []
        row.append(model)
        row = row + body[-1][1:]
        
        newbody.append(row)
            
    #get the max
    row = []
    row.append("max")
    for i in range(1, len(Header)):
        scores = [float(xx[i]) for xx in newbody]
        row.append(numpy.max(scores))
    newbody.append(row)
    
    newname = outputdir + "_".join(models) + ".txt"
    if len(newname) > 50:
        newname = newname[:50] + "_50.txt"
    fio.WriteMatrix(newname, newbody, Header)    

def getSingleCoverage(entries, sources, N):
    covered = []
    
    for entry in entries:
        if entry not in sources:
            print entry
            continue
        covered = covered + sources[entry]

    return len(set(covered))*1.0/N

def getSingleQuality(entries, sources, qualitydict, N):
    scores = []
    for entry in entries:
        if entry not in sources:
            print entry
            continue
        
        score = 0.0
        for id in sources[entry]:
            if id not in qualitydict: continue
            if qualitydict[id] == 'a': continue
            
            score = score + float(qualitydict[id])
        score = score / len(sources[entry])
        
        scores.append(score)

    return numpy.mean(scores)

def getSingleDiversity(entries, sources):
    covered = []
    
    for entry in entries:
        if entry not in sources:
            print entry
            continue
        covered = covered + sources[entry]
        
    covered = set(covered) #get all covered students
    
    #for each of the student, get the probability
    dict = defaultdict(float)
    for entry in entries:
        if entry not in sources:continue
        ids = sources[entry]
        
        for id in ids:
            dict[id] = dict[id] + 1.0
    
    N = len(covered)
    
    for k, v in dict.items():#normailize to probablity
        dict[k] = v/N
        assert(dict[k] <= 1.0)
    
    #get the entropy
    entropy = 0
    for k, v in dict.items(): #normailize to probablity
        entropy = entropy - v * numpy.log(v)
        
    return entropy

def getCoverage(modelname, excelfile, npdir, method="unigram"):
    sheets = range(0,12)
    
    newhead = ['week', 'POI', 'MP', 'LP']
    newbody = []
    
    datadir = "../../mead/data/" + modelname + '/'
    
    header = ['ID', 'Gender', 'Point of Interest', 'Muddiest Point', 'Learning Point']
    summarykey = "Top Answers"
    
    for i, sheet in enumerate(sheets):
        week = i + 1
        
        orig = prData(excelfile, sheet)
        
        row = []
        row.append(week)
        
        for type in ['POI', 'MP', 'LP']:
            path = datadir + str(week)+ '/'
            summaryfile = path + type + '.summary'
            summaries = [line.strip() for line in fio.ReadFile(summaryfile)]
            
            sourcefile = npdir + str(week)+ '/' + type + '.'+method+'.keys.source'
            student_summaryList = getStudentResponseList(orig, header, summarykey, type, withSource=True)
            N = getValidStudentNum(student_summaryList)
            
            print sourcefile, summaryfile
            
            with open(sourcefile, 'r') as infile:
                dict = json.load(infile)
            
            coverage = getSingleCoverage(summaries, dict, N)
            assert(coverage <= 1.0)
            row.append(coverage)
        
        newbody.append(row)
    
    row = []
    row.append("average")
    for i in range(1, len(newhead)):
        scores = [float(xx[i]) for xx in newbody]
        row.append(numpy.mean(scores))
    newbody.append(row)
        
    file = "../data/coverage." + modelname + '.txt'
    fio.WriteMatrix(file, newbody, newhead)

def getDiversity(modelname, excelfile, npdir, method="unigram"):
    sheets = range(0,12)
    
    newhead = ['week', 'POI', 'MP', 'LP']
    newbody = []
    
    datadir = "../../mead/data/" + modelname + '/'
    
    header = ['ID', 'Gender', 'Point of Interest', 'Muddiest Point', 'Learning Point']
    summarykey = "Top Answers"
    
    for i, sheet in enumerate(sheets):
        week = i + 1
        
        orig = prData(excelfile, sheet)
        
        row = []
        row.append(week)
        
        for type in ['POI', 'MP', 'LP']:
            path = datadir + str(week)+ '/'
            summaryfile = path + type + '.summary'
            summaries = [line.strip() for line in fio.ReadFile(summaryfile)]
            
            sourcefile = npdir + str(week)+ '/' + type + '.'+method+'.keys.source'
            student_summaryList = getStudentResponseList(orig, header, summarykey, type, withSource=True)
            N = getValidStudentNum(student_summaryList)
            
            print sourcefile, summaryfile
            
            with open(sourcefile, 'r') as infile:
                dict = json.load(infile)
            
            diversity = getSingleDiversity(summaries, dict)
            row.append(diversity)
        
        newbody.append(row)
    
    row = []
    row.append("average")
    for i in range(1, len(newhead)):
        scores = [float(xx[i]) for xx in newbody]
        row.append(numpy.mean(scores))
    newbody.append(row)
        
    file = "../data/diversity." + modelname + '.txt'
    fio.WriteMatrix(file, newbody, newhead)

def getHighQualityRatio(modelname, excelfile, npdir, method="unigram"):
    sheets = range(0,12)
    
    newhead = ['week', 'MP']
    newbody = []
    
    datadir = "../../mead/data/" + modelname + '/'
    
    header = ['ID', 'Gender', 'Point of Interest', 'Muddiest Point', 'Learning Point']
    summarykey = "Top Answers"
    
    for i, sheet in enumerate(sheets):
        week = i + 1
        
        orig = prData(excelfile, sheet)
        
        row = []
        row.append(week)
        
        for type in ['MP']:
            path = datadir + str(week)+ '/'
            summaryfile = path + type + '.summary'
            summaries = [line.strip() for line in fio.ReadFile(summaryfile)]
            
            qualitydict = getStudentQuality(orig, header)
            
            sourcefile = npdir + str(week)+ '/' + type + '.'+method+'.keys.source'
            student_summaryList = getStudentResponseList(orig, header, summarykey, type, withSource=True)
            N = getValidStudentNum(student_summaryList)
            
            print sourcefile, summaryfile
            
            with open(sourcefile, 'r') as infile:
                dict = json.load(infile)
            
            coverage = getSingleQuality(summaries, dict, qualitydict, N)
            row.append(coverage)
        
        newbody.append(row)
    
    row = []
    row.append("average")
    for i in range(1, len(newhead)):
        scores = [float(xx[i]) for xx in newbody]
        row.append(numpy.mean(scores))
    newbody.append(row)
        
    file = "../data/quality." + modelname + '.txt'
    fio.WriteMatrix(file, newbody, newhead)
            
def getCoverageDiversity(modelname, excelfile, npdir, method="unigram"):
    getCoverage(modelname, excelfile, npdir, method)
    getDiversity(modelname, excelfile, npdir, method)
    getHighQualityRatio(modelname, excelfile, npdir, method)

def getTopRankPhrase(NPs, clusterids, cid, lexdict, sources):
    #get cluster NP, and scores
    dict = {}
    
    s = []
    
    for NP, id, source in zip(NPs, clusterids, sources):
        if int(id) == cid:
            dict[NP] = lexdict[NP.lower()]
            s.append(source)
    
    keys = sorted(dict, key=dict.get, reverse =True)
    
    source = set(s)
    return keys[0], source

def RankCluster2(NPs, lexdict, clusterids, sources):
    sdict = {}
    for id, source in zip(clusterids, sources):
        if id not in sdict:
            sdict[id] = set([])
        sdict[id].add(source)
    
    sizedict = {}
    for key in sdict:
        sizedict[key] = len(sdict[key])
    
    print "sizedict"        
    fio.PrintDict(sizedict)
    
    keys = sorted(sizedict, key=sizedict.get, reverse =True)
    
    return keys

def RankCluster(NPs, lexdict, clusterids, sources):
    sdict = {}
    for id, source in zip(clusterids, sources):
        if id not in sdict:
            sdict[id] = set([])
        sdict[id].add(source)
    
    sizedict = {}
    for key in sdict:
        sizedict[key] = len(sdict[key])

    #get lex scores for clusters
    highestlexdict = {}
    for key in sdict:
        phrase, source = getTopRankPhrase(NPs, clusterids, int(key), lexdict, sources)
        highestlexdict[key] = lexdict[phrase]
    
    print "highestlexdict" 
    fio.PrintDict(highestlexdict)
        
    
    print "sizedict"        
    fio.PrintDict(sizedict)
    
    tkeys = sorted(sizedict, key=sizedict.get, reverse =True)
    
    #break the tires
    keys = []
    N = len(tkeys)
    i = 0
    while i < len(tkeys):
        tkey = []
        j = i
        while (j < N):
            if sizedict[tkeys[j]] == sizedict[tkeys[i]]:
                tkey.append(tkeys[j])
                j = j + 1
            else:
                break
        if j==i:
            i = i + 1
        else:
            i = j
        
        print i
        
        if len(tkey) == 1:
            keys = keys + tkey
        else:
            #sort them
            tdict = {}
            for key in tkey:
                tdict[key] = highestlexdict[key]
            tkey = sorted(tdict, key=tdict.get, reverse =True)
            keys = keys + tkey
        
    assert(len(keys) == len(tkeys))
    
    print keys
    return keys

def PrintExample():
    pass
    
def PrintCluster():
    output = "../data/np511/3/MP.cluster.kmedoids.sqrt.optimumComparerLSATasa.syntax"
    lexfile = "../data/np511/3/MP.syntax.lexrankmax.dict"
    sourcesfile = "../data/np511/3/MP.syntax.source.txt"
    
    import json
    
    body = fio.ReadMatrix(sourcesfile, False)
    NPCandidates = [row[0] for row in body]
    sources = [row[1] for row in body]
    
    
    lexdict = fio.LoadDict(lexfile, 'float')
    
    body = fio.ReadMatrix(output, False)
            
    NPs = [row[0] for row in body]
    clusterids = [row[1] for row in body]
    
    assert(NPCandidates == NPs)
    
    sdict = {}
    for id, source in zip(clusterids, sources):
        if id not in sdict:
            sdict[id] = set([])
        
        sdict[id].add(source)
    
    sizedict = {}
    for key in sdict:
        sizedict[key] = len(sdict[key])
    
    cluster = {}
    for row in body:
        cluster[row[0]] = int(row[1])
    
    Summary = []
    
    keys = RankCluster(NPs, lexdict, clusterids, sources)
    
    clusters = []
    dict = {}
    for key in keys:
        
        #get the phrases
        P = []
        
        centroid = NPs[int(key)]
        
        tdict = {}
        
        for i, (NP, id) in enumerate(zip(NPs, clusterids)):
            #if i == int(key): continue
            if int(id) == int(key):
                tdict[str(i)+"_"+NP] = lexdict[NP]
        
        NPss = sorted(tdict, key=tdict.get, reverse=True)
        
        newNPss = []
        for i, NP in enumerate(NPss):
            k = NP.find("_")
            NP = NP[k+1:]
            
            if i==0:
                NP = NP + '(' + str(lexdict[NP]) + ')'
                
            if NP == centroid:
                NP = "@"+NP
            newNPss.append(NP)
        
        clusters.append(newNPss)
    
    print "cluster", "\t", "student #","\t", "size", "\t","phrases"
    for i, cluster in enumerate(clusters):
        print i+1, '\t', sizedict[keys[i]], '\t', len(cluster), '\t', ', '.join(cluster)

def AllModels():
    basic = "ClusterARank"
    
    models = []
    for i in range(1, 685):
        models.append(basic + str(i))    
    return models
                                          
if __name__ == '__main__':
    excelfile = "../data/2011Spring.xls"
    output = "../data/2011Spring_overivew.txt"
    summaryoutput = "../data/2011Spring_summary.txt"
    
    
    #datadir_multiple = "../../mead/data/2011SpringMutiple/"
    
    #formatedsummary = '../data/2011Spring_Mead_multiplesummary.txt'
    formatedsummary = '../data/2011Spring_Mead_summary.txt'
    TAwordcount = '../data/2011Spring_ta_wordcount.txt'
    
    rougescore = "../data/2011Spring_rouge_single.txt"
    rougescore_multiple = "../data/2011Spring_rouge_multiple.txt"
    
    sennadatadir = "../data/senna/"
    
    npdir = "../data/np/"
    #ExtractUnigramQuality(excelfile, sennadatadir, npdir, 'syntax')
    
#     modelname = 'ShallowSummary_unigram_remove_stop'
#     #getCoverageDiversity(modelname, excelfile, npdir, method = 'unigram')
#     getHighQualityRatio(modelname, excelfile, npdir, method = 'unigram')
#      
#     modelname = 'ShallowSummary_ClusteringNP_KMedoidMalformedKeyphrase_0.6_npsoft_chunk'
#     #getCoverageDiversity(modelname, excelfile, npdir, method = 'chunk')
#     getHighQualityRatio(modelname, excelfile, npdir, method = 'chunk')
#       
#     modelname = 'ShallowSummary_ClusteringNP_KMedoidMalformedKeyphrase_0.2_optimumComparerLSATasa_chunk'
#     #getCoverageDiversity(modelname, excelfile, npdir, method = 'chunk')
#     getHighQualityRatio(modelname, excelfile, npdir, method = 'chunk')
    
    #datadir = '../data/ShallowSummary_unigram_remove_stop.txt'
    #getCoverage(excelfile, npdir, output, method = 'unigram')
    
    getStudentResponseWordCountDistribution2(excelfile, '../data/studentword_distribution.txt')
    
    #ExtractNPSource(excelfile, sennadatadir, outdir, 'syntax')
    #ExtractNPSource(excelfile, sennadatadir, outdir, 'chunk')
    #ExtractUnigramSource(excelfile, outdir)
    #load(excelfile, output)
    #getSummaryOverview(excelfile, summaryoutput)
    
    #Write2Mead(excelfile, datadir)
    #formatSummaryOutput(excelfile, datadir_multiple, output=formatedsummary)
    #formatSummaryOutput(excelfile, datadir, output=formatedsummary)
    #getTAWordCountDistribution(excelfile, TAwordcount)
    #getWordCount(formatedsummary, TAwordcount)
    #getMeadAverageWordCount(formatedsummary, '../data/2011Spring_mead_avaregewordcount.txt')
    #getStudentResponseAverageWords(excelfile, '../data/averageword.txt')
    #getTALengthDistribution(excelfile, '../data/studentword_distribution.txt')
    #getStudentResponseWordCountDistribution(excelfile, '../data/studentword_distribution.txt')
    #GetRougeScore(datadir_multiple, rougescore_multiple)
    #GetRougeScore(datadir = "../../mead/data/", models = ['2011Spring', 'RandombaselineK3', 'RandombaselineK2', 'RandombaselineK1', 'LongestbaselineK3', 'LongestbaselineK2', 'LongestbaselineK1', 'ShortestbaselineK3', 'ShortestbaselineK2', 'ShortestbaselineK1'], outputdir = "../data/" )
    #GetRougeScore(datadir = "../../mead/data/", models = ['TopicWordStem', 'ShallowSummary_bigram_remove_stop','ShallowSummary_weightedngram_remove_stop','ShallowSummary_unigram_remove_stop', 'ShallowSummary_ngram_remove_stop'], outputdir = "../data/" )
    #'ShallowbasedExtrativeSummary_topicS', 'ShallowbasedExtrativeSummary_unigram'
    #ShallowSummary_NPhraseHard ShallowSummary_NPhraseSoft
    #'ShallowSummary_unigram_tfidf'
    #'ShallowSummary_nphard', 'ShallowSummary_npsoft', 'ShallowSummary_greedyComparerWNLin'
    #'ShallowSummary_WeightedgreedyComparerWNLin', 'ShallowSummary_WeightedoptimumComparerWNLin', 'ShallowSummary_WeightedoptimumComparerLSATasa', 'ShallowSummary_WeighteddependencyComparerWnLeskTanim', 'ShallowSummary_WeightedlsaComparer', 'ShallowSummary_WeightedbleuComparer', 'ShallowSummary_WeightedcmComparer', 'ShallowSummary_WeightedlexicalOverlapComparer'
    #'ShallowSummary_AdjNounPhrase_Hard_NoSingleNoun', 'ShallowSummary_AdjNounPhrase_Hard_WithSingleNoun', 'ShallowSummary_AdjNounPhrase_Soft_NoSingleNoun', 'ShallowSummary_AdjNounPhrase_Soft_WithSingleNoun'
    #'ShallowSummary_SyntaxNPhraseHard', 'ShallowSummary_SyntaxNPhraseSoft'
    #'ShallowSummary_ClusteringNPhraseSoft', ShallowSummary_ClusteringSyntaxNPhraseSoft
    #ShallowSummary_ClusteringNP_KMedoid_sqrt_lexicalOverlapComparer
    #ShallowSummary_ClusteringNP_KMedoid_sqrt_npsoft
    #'PhraseMead_chunk', 'PhraseMead_syntax', 'PhraseMead_candidate', 'PhraseMead_candidatestemming'
    
    PrintCluster()
    
    models = [#'ShallowSummary_unigram', #
              #'ShallowSummary_unigram_remove_stop', #'ShallowSummary_unigram_tfidf',
              #'ShallowSummary_bigram',
              #'keyphraseExtractionbasedShallowSummary',
              #'ShallowSummary_NPhrase_chunk_TFIDF', 'ShallowSummary_NPhrase_syntax_TFIDF',
              #'PhraseMead_chunk', 
              #'PhraseMead_syntax', 
              #'PhraseMeadLexRank_chunk', 
              
              #'PhraseMeadLexRank_syntax', 
              #'Clustering_lexrank_sqrt_npsoft_chunk', 'Clustering_lexrank_sqrt_npsoft_syntax', 'Clustering_lexrank_sqrt_optimumComparerLSATasa_chunk', 'Clustering_lexrank_sqrt_optimumComparerLSATasa_syntax',
              #'Clustering_lexrankmax_sqrt_npsoft_chunk', 'Clustering_lexrankmax_sqrt_npsoft_syntax', 'Clustering_lexrankmax_sqrt_optimumComparerLSATasa_chunk', 'Clustering_lexrankmax_sqrt_optimumComparerLSATasa_syntax',
              #'ShallowSummary_ClusteringNP_KMedoid_sqrt_lexicalOverlapComparer'
              #'ShallowSummary_ClusteringNP_KMedoidMalformedKeyphrase_LexRank_sqrt_optimumComparerLSATasa_syntax', 'ShallowSummary_ClusteringNP_KMedoidMalformedKeyphrase_LexRank_sqrt_optimumComparerLSATasa_chunk', 'ShallowSummary_ClusteringNP_KMedoidMalformedKeyphrase_LexRank_sqrt_npsoft_syntax', 'ShallowSummary_ClusteringNP_KMedoidMalformedKeyphrase_LexRank_sqrt_npsoft_chunk',
              
              #'Phrase_syntax_lexrank', 'Phrase_chunk_lexrank',
              #'PhraseLexRankMMR_chunk', 'PhraseLexRankMMR_syntax',
              #'PhraseMead_syntax', 'PhraseMead_chunk', 'PhraseMeadLexRank_syntax', 'PhraseMeadLexRank_chunk',
              #'C4_ShallowSummary_unigram',
              #'C4_ShallowSummary_bigram',
              #'C4_Phrase_syntax_lexrank', 'C4_Phrase_chunk_lexrank',
              #'C4_Clustering_lexrankmax_sqrt_npsoft_chunk', 'C4_Clustering_lexrankmax_sqrt_npsoft_syntax', 'C4_Clustering_lexrankmax_sqrt_optimumComparerLSATasa_chunk', 'C4_Clustering_lexrankmax_sqrt_optimumComparerLSATasa_syntax',
              
              #"C4_keyphraseExtractionbasedShallowSummary",
              #"ClusteringAlone",
              #"ClusterARank",
              #'C4_Clustering_lexrankmax_sqrt_optimumComparerLSATasa_syntax',
              #'C4_Clustering_lexrankmax_4_npsoft_chunk', 'C4_Clustering_lexrankmax_4_npsoft_syntax', 'C4_Clustering_lexrankmax_4_optimumComparerLSATasa_chunk', 'C4_Clustering_lexrankmax_4_optimumComparerLSATasa_syntax',         
              
              #'ShallowSummary_unigram_remove_stop_K6', 'ShallowSummary_bigram_K6', 'C6_keyphraseExtractionbasedShallowSummary', 'C6_PhraseMead_syntax', 'C6_PhraseMeadMMR_syntax', 'C6_PhraseMeadLexRank_syntax', 'C6_PhraseMeadLexRankMMR_syntax', 'C6_LexRank_syntax', 'C6_LexRankMMR_syntax','C6_ClusterARank511',
            'C4_unigram_remove_stop', 'C4_bigram', 'C4_keyphrase', "C4_Mead", 
            'C4_PhraseMead_syntax', 'C4_PhraseMeadMMR_syntax',
            'C4_PhraseMeadLexRank_syntax',
            'C4_PhraseMeadLexRankMMR_syntax', 
            'C4_LexRank_syntax', 'C4_LexRankMMR_syntax',
            "C4_ClusteringAlone", 
            'C4_ClusterARank511',
            'Opinosis',
            'OpinosisPhrase',
            'OpinosisPhraseCluster'
            ]
    
    #models = AllModels()
    
    GetRougeScore(datadir = "../../mead/data/", models = models, outputdir = "../data/" )
    CombineRouges(models = models, outputdir = "../data/")
    
    models = ['C4_PhraseMeadLexRankMMR_syntax', 
              #'PhraseMeadMMR_syntax', 
              #'PhraseMeadMMR_chunk'
              ]
    #GetRougeScoreMMR(datadir = "../../mead/data/", models = models, outputdir = "../data/")
       
    #GetRougeScore(datadir = "../../mead/data/", model = "2011Spring", outputdir = "../data/" )
    #GetRougeScore(datadir, rougescore)
    #TASummaryCoverage(excelfile, output="../data/coverage.txt")
    #print getNgram("1 2 3 4 5 6", 6)

    
#     datadir = "../../mead/data/ShallowSummary_SyntaxNPhraseSoft/"
#     outdir = "../data/np/"
#     ExtractNP(datadir, outdir, 'syntax')
#    
    
    #methods = ['npsoft', 'greedyComparerWNLin', 'optimumComparerLSATasa','optimumComparerWNLin',  'dependencyComparerWnLeskTanim', 'lexicalOverlapComparer']
#     methods = ['optimumComparerLSATasa']
#     ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
#     nps = ['syntax']
#     #ShallowSummary_ClusteringNP_KMedoid, ShallowSummary_ClusteringNP_KMedoidMalformedKeyphrase
#     for ratio in ratios:
#         for method in methods: #'bleuComparer', 'cmComparer', 'lsaComparer',
#             for np in nps:
#                 GetRougeScore(datadir = "../../mead/data/", models = ['C4_Clustering_lexrankmax' + "_"+ str(ratio)+"_"+method+"_"+np], outputdir = "../data/" )
#         
#     CombineKMethod("../data/", "../data/C4_Clustering_lexrankmax.txt", methods, ratios, nps, 'C4_Clustering_lexrankmax')
#    
  
    print "done"