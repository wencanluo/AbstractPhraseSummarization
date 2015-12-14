import fio
import os
import json
import re
import xml.etree.ElementTree as ET
import NLTKWrapper

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
        for doc_id in data[folder]:
            path = os.path.join(outdir, folder)
            fio.NewPath(path)
            
            filename = os.path.join(path, doc_id+'.key')
            
            allsentences = []
            for doc in data[folder][doc_id]['docs']:
                sentences = ParseTACXML(doc)
                allsentences += sentences
            
            fio.SaveList(allsentences, filename)

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
        
        
        row = [document_num, sentence_num, sentence_num/document_num, word_num/document_num]
        body.append(row)
    
    header = ['# of document',   '# of sentences',    'average # of sentences',    'average # of words']
    fio.WriteMatrix("../../data/TAC/statistics.txt", body, header)
    
def ExtractReferenceSummary(datadir, outdir):
    fio.NewPath(outdir)
    
    filelist = datadir+'list.json'
    data = fio.LoadDictJson(filelist)
    
    for folder in data:
        for doc_id in data[folder]:
            path = os.path.join(outdir, folder)
            fio.NewPath(path)
            
            models = data[folder][doc_id]['models']
            
            assert(len(models) == 4)
            
            for i, doc in enumerate(models):
                sentences = fio.LoadList(doc)
                filename = os.path.join(path, doc_id+ '.ref%d.summary'%(i+1))
                print filename
                
                fio.SaveList(sentences, filename)
            
if __name__ == '__main__':
    datadir = "../../data/DUC04/"
    
    outdirs = [#"../../data/DUC_ILP_Sentence/",
               #'../../data/DUC_MC/',
               '../../data/DUC_ILP_MC/',
               ]
    
    for outdir in outdirs:
        ExtractSentence(datadir, outdir)
        ExtractReferenceSummary(datadir, outdir)
        
    #ExtractDataStatitics(datadir, outdir)
    
    
    #ss = ParseTACXML('../../data/TAC/s10\\test_doc_files\\D1035G\\D1035G-A\\new\\APW19980718.0011')
   