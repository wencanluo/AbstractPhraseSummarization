import fio
import numpy
import postProcess
from OrigReader import prData
import Survey
import NLTKWrapper

def getAverageWordLength(excelfile):
    header = ['ID', 'Gender', 'Point of Interest', 'Muddiest Point', 'Learning Point']
    summarykey = "Top Answers"
    
    sheets = range(0, 12)
    
    wc = []
    
    for i, sheet in enumerate(sheets):
        week = i + 1
        orig = prData(excelfile, sheet)
        
        for type in ['POI', 'MP', 'LP']:
            student_responseList = Survey.getStudentResponseList(orig, header, summarykey, type, withSource=False, Split=False)
            
            for response in student_responseList:
                if len(response.strip().split()) <= 2:
                       print response
                wc.append(len(response.strip().split()))

    print numpy.max(wc), numpy.mean(wc), numpy.min(wc), numpy.std(wc)
    
    
def GetTable1forENMLP(excelfile):
    header = ['ID', 'Gender', 'Point of Interest', 'Muddiest Point', 'Learning Point']
    summarykey = "Top Answers"
    
    sheets = range(0, 12)
        
    total_sentence = {'POI':[], 'MP':[], 'LP':[]}
    total_words = {'POI':[], 'MP':[], 'LP':[]}
    sentence_length  = {'POI':[], 'MP':[], 'LP':[]}
    
    total_words_ta = {'POI':[], 'MP':[], 'LP':[]}
    total_bigram_ta = {'POI':[], 'MP':[], 'LP':[]}
    total_covered_bigram_ta = {'POI':[], 'MP':[], 'LP':[]}
    
    for i, sheet in enumerate(sheets):
        week = i + 1
        orig = prData(excelfile, sheet)
        
        for type in ['POI', 'MP', 'LP']:
            student_responseList = Survey.getStudentResponseList(orig, header, summarykey, type, withSource=False, Split=True)
            
            sc = len(student_responseList)
            total_sentence[type].append(sc)
            
            wc = 0.0
            for response in student_responseList:
                wc += len(response.split())
            total_words[type].append(wc)
            
            sentence_length[type].append( wc / sc )
            
            summaryList = Survey.getTASummary(orig, header, summarykey, type)
            
            wc_ta = 0
            bigram_ta = 0
            covered_bigram_ta = 0
            for summary in summaryList:
                wc_ta += len(summary.split())
                
                bigrams = NLTKWrapper.getNgram(summary, 2)
                bigram_ta += len(bigrams)
                for token in bigrams:
                    if postProcess.CheckKeyword(token, student_responseList):
                        covered_bigram_ta += 1
                            
            total_words_ta[type].append(wc_ta)
            total_bigram_ta[type].append(bigram_ta)
            total_covered_bigram_ta[type].append(covered_bigram_ta)
    
    body = []
    row = []
    for type in ['POI', 'MP', 'LP']:
        row.append(numpy.mean(total_sentence[type]))
    body.append(row)
    
    sentences_total = []
    for type in ['POI', 'MP', 'LP']:
        sentences_total += total_sentence[type]
    print 'average number of sentence:', numpy.mean(sentences_total)
    
    row = []
    for type in ['POI', 'MP', 'LP']:
        row.append(numpy.mean(sentence_length[type]))
    body.append(row)
    
    row = []
    for type in ['POI', 'MP', 'LP']:
        row.append(numpy.mean(total_words[type]))
    body.append(row)
    
    words_total = []
    for type in ['POI', 'MP', 'LP']:
        words_total += total_words[type]
    print 'average number of words:', numpy.mean(words_total)
    
    row = []
    for type in ['POI', 'MP', 'LP']:
        row.append(numpy.mean(total_words_ta[type]))
    body.append(row)
    
    row = []
    for type in ['POI', 'MP', 'LP']:
        row.append(numpy.mean(total_bigram_ta[type]))
    body.append(row)
    
    row = []
    for type in ['POI', 'MP', 'LP']:
        row.append(numpy.mean(total_covered_bigram_ta[type]))
    body.append(row) 
    
    fio.WriteMatrix("../../data/statistics.txt", body, header=None)   
    

if __name__ == '__main__':
    
    excelfile = "../../data/2011Spring_norm.xls"
    
    GetTable1forENMLP(excelfile)
    #getAverageWordLength(excelfile)
                                           