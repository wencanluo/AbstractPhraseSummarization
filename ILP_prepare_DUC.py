import fio
import os
import json
import re
import xml.etree.ElementTree as ET
import NLTKWrapper
from collections import defaultdict

def ParseTACXML(input):
    lines = fio.ReadFile(input)
    
    paragraphs = []
    
    flag = False
    
    paragraph = []
    for line in lines:
        if line.startswith('<TEXT>'):
            flag = True
            paragraph = []
            continue
        
        if line.startswith('</TEXT>'):
            paragraphs.append(' '.join(paragraph))
            flag = False
            continue
        
        if flag:
            paragraph.append(line.strip())
        
    lstSentences = []
    for paragraph in paragraphs:
        sentences = NLTKWrapper.splitSentence(paragraph)
        lstSentences += sentences
    return lstSentences
           
def ExtractSentence(datadir, outdir):
    fio.NewPath(outdir)
    
    filelist = datadir+'list.json'
    data = fio.LoadDictJson(filelist)
    
    for folder in data:
        for i, doc_id in enumerate(data[folder]):
            path = os.path.join(outdir, str(i))
            fio.NewPath(path)
            
            source = {}
            responses = []
            count = defaultdict(int)
            
            for doc in data[folder][doc_id]['docs']:
                sentences = ParseTACXML(doc)
                
                for sentence in sentences:
                    source[sentence] = doc_id
                    count[sentence] += 1
                    
                responses += sentences
            
            type= 'q1'
            
            outout = os.path.join(path, type + ".sentence.key")
            fio.SaveList(set(responses), outout)
             
            output = os.path.join(path, type + '.sentence.keys.source')
            fio.SaveDict2Json(source, output)
             
            output = os.path.join(path, type + '.sentence.dict')
            fio.SaveDict(count, output)
            

def ExtractDataStatitics(datadir, outdir):
    fio.NewPath(outdir)
    
    filelist = datadir+'list.json'
    data = fio.LoadDictJson(filelist)
    
    body = []
    for folder in data:
        document_num = len(data[folder])
        
        sentence_num = 0.0
        word_num = 0.0
        
        for doc_id in data[folder]:
            path = os.path.join(outdir, folder)
            fio.NewPath(path)
            
            filename = os.path.join(path, doc_id+'.key')
            
            allsentences = []
            for doc in data[folder][doc_id]['docs']:
                sentences = ParseTACXML(doc)
                allsentences += sentences
                
                for sentence in sentences:
                    word_num += len(sentence.split())
            
            fio.SaveList(allsentences, filename)
            sentence_num += len(allsentences)
            
        row = [folder, document_num, sentence_num, sentence_num/document_num, word_num/document_num]
        body.append(row)
    
    header = ['data', '# of document',   '# of sentences',    'average # of sentences',    'average # of words']
    fio.WriteMatrix(os.path.join(outdir, "statistics.txt"), body, header)
    
def ExtractReferenceSummary(datadir, outdir):
    fio.NewPath(outdir)
    
    filelist = datadir+'list.json'
    data = fio.LoadDictJson(filelist)
    
    for folder in data:
        for lec, doc_id in enumerate(data[folder]):
            path = os.path.join(outdir, str(lec))
            fio.NewPath(path)
            
            models = data[folder][doc_id]['models']
            
            type= 'q1'
            
            assert(len(models) == 4)
            
            for i, doc in enumerate(models):
                sentences = fio.LoadList(doc)
                filename = os.path.join(path, type+ '.ref.%d'%(i))
                print filename
                
                fio.SaveList(sentences, filename)
            
if __name__ == '__main__':
    #ExtractDataStatitics('../../data/TAC/', '../../data/TAC/')
#     ExtractDataStatitics('../../data/DUC/DUC04/', '../../data/DUC/DUC04/')
#     exit(-1)
    
    datadir = "../../data/DUC/DUC04/"
    
    outdirs = ['../../data/DUC04/MC/',
               '../../data/DUC04/ILP_MC/',
               ]
    
    
    for outdir in outdirs:
        ExtractSentence(datadir, outdir)
        ExtractReferenceSummary(datadir, outdir)
    
    #ss = ParseTACXML('../../data/TAC/s10\\test_doc_files\\D1035G\\D1035G-A\\new\\APW19980718.0011')
   