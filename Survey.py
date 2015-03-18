from OrigReader import prData
import sys
import re
import fio
import NLTKWrapper
import MaximalMatchTokenizer
import numpy as np
import math
import SennaParser

stopwordfilename = "../../../Fall2014/summarization/ROUGE-1.5.5/data/smart_common_words.txt"
filters = ["?", "[blank]", 'n/a', 'blank'] #a classifier to predict whether the student has problem
#filters = []
    
spellcheckerfile = "../../data/spellchecker.txt"
spellchecker = fio.LoadDict(spellcheckerfile)
                    
def HasSummary(orig, header, summarykey):
    key = header[0]
    for k, inst in enumerate(orig._data):
        try:
            value = inst[key].lower().strip()
            if value == summarykey.lower():
                if len(inst[header[2]].strip()) > 0:
                    return True
        except Exception:
            return False
    return False

def getWeight(summary):
    summary = summary.strip()
    g = re.search('^\d+\).*\[(\d+)\]$', summary)
    if g != None:
        weight = g.group(1).strip()
        return int(weight)
    
    g = re.search('^.*\[(\d+)\]$', summary)
    if g != None:
        weight = g.group(1).strip()
        return int(weight)
    
    return 1


def NormalizedResponse(s):
    s = s.strip()
    
    punctuations = ".?!:;-()[]/,"
    
    if len(s) > 0:
        if s[0] in punctuations:
            s = s[1:].strip()
        if len(s) > 0:
            if s[0] in punctuations:
                s = s[1:].strip()
    
    if len(s) > 0 and s[-1] not in ['.', '?']:
        s = s + "."
                
    #process the spell error
    unigram = NLTKWrapper.getNgram(s, 1)
    
    newS = []
    for word in unigram:
        if word.lower() in spellchecker:
            word = spellchecker[word.lower()]
        
        newS.append(word)
    
    return " ".join(newS)

def NormalizedTASummary(summary):
    summary = summary.strip()
    g = re.search('^\d+\)(.*)\[\d+\]$', summary)
    if g != None:
        summary = g.group(1).strip()
    
    g = re.search('^(.*)\[\d+\]$', summary)
    if g != None:
        summary = g.group(1).strip()
    
    g = re.search('^\d+\)(.*)$', summary)
    if g != None:
        summary = g.group(1).strip()
    
    return summary.strip()

def NormalizeMeadSummary(summary):
    summary = summary.strip()
    g = re.search('^\[\d+\](.*)$', summary)
    if g != None:
        summary = g.group(1).strip()
    
    return summary.strip()

def getTASummary(orig, header, summarykey, type='POI', weight = False):
    '''
    Get TA's summary from the excel
    return a list of sentences
    '''
    if not HasSummary(orig, header, summarykey):
        return []
    
    if type=='POI':
        keyword = header[2]
    elif type=='MP':
        keyword = header[3]
    elif type=='LP':
        keyword = header[4]
    else:
        return []
    
    key = header[0]
    
    weights = []
    summary = []
    for k, inst in enumerate(orig._data):
        value = inst[key].lower().strip()
        if value == summarykey.lower():
            value = inst[keyword].strip()
            summaries = value.split("\n")
            for sum in summaries:
                w = getWeight(sum)
                summary.append(NormalizedTASummary(sum))
                weights.append(w)
    if weight:
        return summary, weights
    return summary

def WriteTASummary(excelfile, datadir):
    reload(sys)
    sys.setdefaultencoding('utf8')
    
    summarykey = "Top Answers"
    header = ['ID', 'Gender', 'Point of Interest', 'Muddiest Point', 'Learning Point']
    sheets = range(0,12)
    types = ['POI', 'MP', 'LP']
    
    for sheet in sheets:
        week = sheet + 1
        path = datadir + str(week)+ '/'
        fio.NewPath(path)

        orig = prData(excelfile, sheet)
        for type in types:
            summary, weight = getTASummary(orig, header, summarykey, type, weight=True)
    
            #summary = ["ssssss " + str(w) + " " + s for s, w in zip(summary, weight)]
                             
            filename = path + type + '.ref.summary'
            print filename
            
            #only save the first 3 points
            fio.SaveList(summary, filename)

def getStudentQuality(orig, header):
    '''
    return a dictionary of the students' summary, with the student id as a key
    The value is a list with each sentence an entry
    '''
    dict = {}
    
    key = "Muddiest Point Coding (0-3; a = \"I understood everything\")"
        
    for k, inst in enumerate(orig._data):
        try:
            value = inst['ID'].lower().strip()
            if value == 'top answers': continue
            
            if len(value) > 0:
                content = inst[key]
                dict[value] = content
            else:
                break
        except Exception:
            return dict
    return dict
           
def getStudentResponse(orig, header, summarykey=None, type='POI'):
    '''
    return a dictionary of the students' summary, with the student id as a key
    The value is a list with each sentence an entry
    '''
    summaries = {}
    
    if type=='POI':
        key = header[2]
    elif type=='MP':
        key = header[3]
    elif type=='LP':
        key = header[4]
    else:
        return None
        
    for k, inst in enumerate(orig._data):
        try:
            value = inst['ID'].lower().strip()
            if value == 'top answers': continue
            
            if len(value) > 0:
                content = inst[key].strip()
                
                if content.lower() in filters: continue
                
                if len(content) > 0:
                    summary = NLTKWrapper.splitSentence(content)
                    summaries[value] = summary
            else:
                break
        except Exception:
            return summaries
    return summaries

def getMPQualityPoint(orig):
    '''
    return a list of the students' muddiest point with scores
    '''
    summaries = []
    
    key = "Muddiest Point"
    pointkey = "Muddiest Point Coding (0-3; a = \"I understood everything\")"
    
    for k, inst in enumerate(orig._data):
        
        if pointkey not in inst:
            continue
            
        try:
            value = inst['ID'].lower().strip()
            if value == 'top answers': continue
            
            if len(value) > 0:
                content = inst[key].strip()
                score = inst[pointkey]
                
                summaries.append((content, score))
            else:
                break
        except Exception:
            return summaries
    return summaries

def getStudentResponseList(orig, header, summarykey, type='POI', withSource=False, Split=True):
    student_summaries = getStudentResponse(orig, header, summarykey, type)
    student_summaryList = []
    
    for id, summaryList in student_summaries.items():
        summaries = []
        for s in summaryList:
            s = NormalizedResponse(s)
            if len(s) == 0: continue
            
            summaries.append(s)
                
        if Split:
            for s in summaries:
                student_summaryList.append((s,id))
        else:
            student_summaryList.append((' '.join(summaries), id))
    
    if withSource:
        return student_summaryList
    else:
        return [summary[0] for summary in student_summaryList]
                      
def getStudentSummaryNum(orig, header, summarykey, type='POI'):
    if type=='POI':
        key = header[2]
    elif type=='MP':
        key = header[3]
    elif type=='LP':
        key = header[4]
    else:
        return 0
    
    count = 0
    for k, inst in enumerate(orig._data):
        try:
            value = inst['ID'].lower().strip()
            if len(value) > 0:
                if len(inst[key].lower().strip()) > 0:
                    count = count + 1
            else:
                break
        except Exception:
            return 0
    return count-1

def getMaleNum(orig, header, summarykey):
    key = header[1]
    count = 0
    for k, inst in enumerate(orig._data):
        try:
            value = inst[key].lower().strip()
            if value == 'm':
                count = count + 1
        except Exception:
            return 0
    return count

def getStudentNum(orig, header, summarykey):
    key = header[0]
    count = 0
    for k, inst in enumerate(orig._data):
        try:
            value = inst[key].lower().strip()
            if len(value) > 0:
                count = count + 1
            else:
                break
        except Exception:
            return 0
    return count-1

def getValidStudentNum(student_summaryList):
    ids = []
    for _, id in student_summaryList:
        ids.append(id)
    return len(set(ids))

def getMeadSummary(datadir, type):
    #return a list of summaries, week by week. The summary for each week is also a list
    sheets = range(0,12)
    
    summaries = []
    
    for sheet in sheets:
        row = []
        week = sheet + 1
        row.append(week)
                    
        path = datadir + str(week)+ '/'
        filename = path + type + '.summary'
        
        if not fio.isExist(filename): continue
        
        lines = fio.readfile(filename)
        summary = []
        for line in lines:
            summary.append(NormalizeMeadSummary(line))
        
        summaries.append(summary)
    
    return summaries
        
def getMeadSummaryList(datadir, type):
    #return a list of summaries, week by week. The summary for each week is also a list
    sheets = range(0,12)
    
    summaries = []
    for sheet in sheets:
        row = []
        week = sheet + 1
        row.append(week)
                    
        path = datadir + str(week)+ '/'
        filename = path + type + '.summary'
        
        lines = fio.readfile(filename)
        summary = []
        for line in lines:
            summary.append(NormalizeMeadSummary(line))
        
        summaries.append(summary)
    
    summaryList = []
    for summaries in summaries:
        for summary in summaries:
            summaryList.append(summary)
        
    return summaryList

def getStudentResponses4Senna(excelfile, datadir, Split=True):
    header = ['ID', 'Gender', 'Point of Interest', 'Muddiest Point', 'Learning Point']
    summarykey = "Top Answers"
    
    sheets = range(0,25)
    #sheets = range(0,12)
    
    for i, sheet in enumerate(sheets):
        week = i + 1
        orig = prData(excelfile, sheet)
        
        for type in ['POI', 'MP', 'LP']:
            student_summaryList = getStudentResponseList(orig, header, summarykey, type, withSource=False, Split=Split)
            filename = datadir + "senna." + str(week) + "." + type + ".input"
            
            #fio.SaveList(student_summaryList, filename + ".2")
            #student_summaryList = [summary[0] for summary in student_summaryList]
            fio.SaveList(student_summaryList, filename)
            
def getStudentResponses4Maui(excelfile, datadir):
    header = ['ID', 'Gender', 'Point of Interest', 'Muddiest Point', 'Learning Point']
    summarykey = "Top Answers"
    
    #sheets = range(0,25)
    sheets = range(0,12)
    
    for i, sheet in enumerate(sheets):
        week = i + 1
        orig = prData(excelfile, sheet)
        
        for type in ['POI', 'MP', 'LP']:
            student_summaryList = getStudentResponseList(orig, header, summarykey, type)
            filename = datadir + "" + str(week) + "." + type + ".txt"
            fio.SaveList(student_summaryList, filename, ".\n")

def getCandidatePhrases(dir):
    sheets = range(0,12)
    
    numbers = []
    
    length = []
    
    for i, sheet in enumerate(sheets):
        week = i + 1
        filename = dir + str(week) + ".txt"
        
        phrases = fio.readfile(filename)
        #print len(phrases)
        numbers.append(len(phrases))
        for phrase in phrases:
            n = len(phrase.split())
            if n>10:
                print phrase
            length.append(len(phrase.split()))
    
    #fio.PrintList(numbers, "\n")
    fio.PrintList(length, ",")

def getLength(NPs):
    N = 0.0
    for NP in NPs:
        N = N + len(NP.split())
    return N / len(NPs)
    
def getNumberofCandiatePhrase(excel, phrasedir):
    header = ['ID', 'Gender', 'Point of Interest', 'Muddiest Point', 'Learning Point']
    summarykey = "Top Answers"
    
    #sheets = range(0,25)
    sheets = range(0,12)
    
    localheader = ['ratio  phrase/sen', 'total phrase', '# word/phrase']
    nhead = ['type',  'Week', 'total sen']
    
    nbody = []
    methodNames = ['Hard Constraint', "Stemming Constraint", "Chunk", "Syntax"]
    methods = [1, 2, 3, 4]
    for type in ['POI', 'MP', 'LP']:
        for method in methods:
            localbody = []
            for i, sheet in enumerate(sheets):
                row = []                
                week = i + 1
                
                orig = prData(excelfile, sheet)
            
                print excelfile, sheet, type
                student_summaryList = getStudentResponseList(orig, header, summarykey, type)
                
                phrasefile = phrasedir + str(week) + ".txt"
                sennadatadir = "../data/senna/"
                sennafile = sennadatadir + "senna." + str(week) + "." + type + '.output'
                sentences = SennaParser.SennaParse(sennafile)
                
                #row.append(len(student_summaryList))
                
                if method == 1 or method == 2:
                    sentenceList = student_summaryList
                else:
                    sentenceList = sentences
                    
                #candiate no stemming
                counts = []
                lengths = []
                for s in sentenceList:
                    if method == 1:
                        NPs = MaximalMatchTokenizer.MaximalMatchTokenizer(s, phrasefile, stemming=False)
                    elif method == 2:
                        NPs = MaximalMatchTokenizer.MaximalMatchTokenizer(s, phrasefile)
                    elif method == 3:
                        NPs = s.getNPrases()
                    elif method == 4:
                        NPs = s.getSyntaxNP()
                    
                    counts.append(len(NPs))
                    if len(NPs) > 0:
                        l = getLength(NPs)
                        lengths.append(l)
                    
                row.append("%.3f" % np.mean(counts))
                row.append(np.sum(counts))
                row.append("%.3f" % np.mean(lengths))
                       
                localbody.append(row)
            
            #get the average
            newrow = []
            newrow.append(methodNames[method-1])
            newrow.append(type)
            for i in range(len(localheader)):
                values = [ float(row[i]) for row in localbody]
                newrow.append("%.3f" % np.mean(values))
            nbody.append(newrow)
            
    fio.writeMatrix("../data/numberofcandiatephrase.txt", nbody, ["method", "type"] + localheader)
                                           
if __name__ == '__main__':
    
    excelfile = "../data/2011Spring.xls"
    datadir = "../../Maui1.2/data/2011Spring/"
    sennadir = "../data/senna/"
    
    #fio.NewPath(datadir)
    #getStudentResponses4Maui(excelfile, datadir)
    
    #getStudentResponses4Senna(excelfile, sennadir)
    
    #getCandidatePhrases("../data/phrases/")
    
    getNumberofCandiatePhrase(excelfile, "../data/phrases/")