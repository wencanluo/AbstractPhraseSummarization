import sys
import re
import fio
import json
import NLTKWrapper
import os
from collections import defaultdict

filters = ["?", "[blank]", 'n/a', 'blank',] #'none', "no", "nothing"

import datetime

RatingKey = {"slightly": 1, 
"somewhat": 2,
"moderately": 3, 
"mostly":4,
"completely":5
}

RateSplitTag = "||Rating: "

stopwordfilename = "../../data/smart_common_words.txt"
stopwords = [line.lower().strip() for line in fio.ReadFile(stopwordfilename)]
punctuations = ['.', '?', '-', ',', '[', ']', '-', ';', '\'', '"', '+', '&', '!', '/', '>', '<', ')', '(', '#', '=']

stopwords = stopwords + punctuations

def getRatingkey(rate):
    key = rate.strip().lower()
    if key in RatingKey:
        return RatingKey[key]
    return -1

def NormalizeResponse(response):
    k = response.find(RateSplitTag)
    if k == -1:
        return response
    return response[:k]
            
def getStudentResponse(excelfile, cid, lecture_number, type):
    '''
    return a dictionary of the students' summary, with the student id as a key
    The value is a list with each sentence an entry
    '''
    f = open(excelfile)
    reflections = json.load(f)['results']
    f.close()
    
    tokenIndex = 'user'
    couseIndex = 'cid'
    lectureIndex = 'lecture_number'
    
    summaries = {}
    
    key = type
        
    for k, inst in enumerate(reflections):
        try:
            token = inst[tokenIndex].lower().strip()
            courseNow = inst[couseIndex].strip()
            lecture = inst[lectureIndex]
            
            if courseNow != cid: continue
            if lecture_number != lecture: continue
            
            if len(token) > 0:
                content = inst[key].strip()
                if content.lower() in filters: continue
                if len(content) > 0:   
                    content = NormalizeResponse(content)                 
                    summary = NLTKWrapper.splitSentence(content)
                    
                    summary = [s.strip() for s in summary]
                    
                    if token in summaries:
                        summaries[token] += summary
                    else:
                        summaries[token] = summary
            else:
                break
        except Exception as e:
            print e
            return summaries
    return summaries

def getStudentResponseList(excelfile, cid, lecture, type, withSource=False):
    student_summaries = getStudentResponse(excelfile, cid, lecture, type)
    student_summaryList = []
    
    for id, summaryList in student_summaries.items():
        for s in summaryList:
            student_summaryList.append((s,id))
            
    if withSource:
        return student_summaryList
    else:
        return [summary[0] for summary in student_summaryList]
    
def getStudentResponses4Senna(excelfile, cid, maxWeek, datadir):
    sheets = range(1, maxWeek+1)
    
    for sheet in sheets:
        week = sheet

        for type in ['q1', 'q2', 'q3', 'q4']:
            student_summaryList = getStudentResponseList(excelfile, cid, week, type)
            if len(student_summaryList) == 0: continue
            
            filename = datadir + "senna." + str(week) + "." + type + ".input"
            
            fio.SaveList(student_summaryList, filename)

def getStudentResponses4Annotation(excelfile, cid, maxWeek, datadir):
    sheets = range(1, maxWeek+1)
    
    for sheet in sheets:
        week = sheet
        
        for type in ['q1', 'q2', 'q3', 'q4']:
            head = ['student_id', 'sentence_id', 'responses']
            body = []
        
            student_summaryList = getStudentResponseList(excelfile, cid, week, type, True)
            
            if len(student_summaryList) == 0: continue
            
            filename = datadir + "response." + str(week) + "." + type + ".txt"
            
            old = ""
            i = 1
            for summary,id in student_summaryList:
                row = []
                summary = summary.replace('"', '\'')
                if len(summary.strip()) == 0: continue
                
                if id == old:
                    row.append(" ")
                else:
                    row.append(id)
                row.append(i)
                row.append(summary)
                body.append(row)
                i = i + 1
                old = id
                
            fio.WriteMatrix(filename, body, head)

def getStudentResponses4Quality(excelfile, cid, maxWeek, datadir):
    sheets = range(1, maxWeek+1)
    
    for sheet in sheets:
        week = sheet
        
        for type in ['q1', 'q2', 'q3', 'q4']:
            head = ['student_id', 'responses']
            body = []
        
            student_summaries = getStudentResponse(excelfile, cid, week, type)
            if len(student_summaries) == 0: continue
            
            for id, summaryList in student_summaries.items():
                summary = ' '.join(summaryList)
                
                row = []
                summary = summary.replace('"', '\'')
                if len(summary.strip()) == 0: continue
                
                row.append(id)
                row.append(summary)
                body.append(row)
            
            filename = datadir + "response." + str(week) + "." + type + ".txt"
            fio.WriteMatrix(filename, body, head)
                   
if __name__ == '__main__':    
    cid = sys.argv[1]
    maxWeek = int(sys.argv[2])
    
    excelfile = "../data/CourseMirror/reflections.json"
    annotation_dir = "../data/Annotation/" + cid + '/'
    sennadir = "../data/"+cid+"/senna/"
    fio.NewPath(sennadir)
    getStudentResponses4Senna(excelfile, cid, maxWeek, sennadir)
    
    fio.NewPath(annotation_dir)
    getStudentResponses4Annotation(excelfile, cid, maxWeek, annotation_dir)
    
    quality_dir = "../data/" + cid + '/quality/'
    fio.NewPath(quality_dir)
    getStudentResponses4Quality(excelfile, cid, maxWeek, quality_dir)
    
    