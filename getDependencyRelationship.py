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
                  
def getInput4StanfordParser(excelfile, spellchecker, outputdir):
    reload(sys)
    sys.setdefaultencoding('utf8')
    
    #load spell check ditionary
    dict = fio.LoadDict(spellchecker)
    
    header = ['ID', 'Gender', 'Point of Interest', 'Muddiest Point', 'Learning Point']
    summarykey = "Top Answers"
    
    sheets = range(0,12)
    
    datahead = ['Week', '# of sentence POI', '# of sentence POI', '# of sentence POI']
    
    for sheet in sheets:
        row = []
        week = sheet + 1
        
        orig = prData(excelfile, sheet)
        
        for type in ['POI', 'MP', 'LP']:
            summaryList = getStudentResponseList(orig, header, summarykey, type)
            
            NormalizedsummaryList = []
            
            for summary in summaryList:
                
                #process the punctuation
                summary = summary.strip()
                if len(summary) > 0 and summary[-1] not in ['.', '?']:
                    summary = summary + "."
                
                #process the spell error
                unigram = NLTKWrapper.getNgram(summary, 1)
                
                newS = []
                for word in unigram:
                    #word = word.lower()
                    
                    if word.lower() in dict:
                        word = dict[word.lower()]
                    
                    newS.append(word)
                
                NormalizedsummaryList.append(" ".join(newS))
            
            filename = outputdir + str(week) + "_" + str(type) + ".input"
            print filename
            
            fio.SaveList(NormalizedsummaryList, filename)
        
if __name__ == '__main__':
    excelfile = "../../data/2011Spring_norm.xls"
    stanford_dir =  "../../data/parser/"
    spellchecker = '../../data/spellchecker.txt'
    
    getInput4StanfordParser(excelfile, spellchecker, stanford_dir)
    
    print "done"