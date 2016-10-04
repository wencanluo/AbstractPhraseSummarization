import fio
import os
import json
import re
import xml.etree.ElementTree as ET
import NLTKWrapper

from collections import defaultdict

def ParseTACXML_notwork(input):
    try:
        tree = ET.parse(input)
        root = tree.getroot()
    
        paragraphs = root.findall('P')

        for paragraph in paragraphs:
            print paragraph.text
    except Exception as e:
        print e
        print input
 
def ParseTACXML(input):
    lines = fio.ReadFile(input)
    
    paragraphs = []
    
    flag = False
    
    paragraph = []
    for line in lines:
        if line.startswith('<P>'):
            flag = True
            paragraph = []
            continue
        
        if line.startswith('</P>'):
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
           
def ExtractSentence(datadir, outdir, year = 's08', split=False):
    filelist = datadir+'list.json'
    data = fio.LoadDictJson(filelist)
    
    folder = year
    
    for lec, doc_id in enumerate(data[folder]):
        
        if split:
            if lec < len(data[folder])/2:
                suboutdir = outdir.replace(year, year+'_A')
            else:
                suboutdir = outdir.replace(year, year+'_B')
                lec = lec/2
        else:
            suboutdir = outdir
            
        path = os.path.join(suboutdir, str(lec))
        fio.NewPath(path)
        
        type = 'q1'
        
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
        
        
        row = [document_num, sentence_num, sentence_num/document_num, word_num/document_num]
        body.append(row)
    
    header = ['# of document',   '# of sentences',    'average # of sentences',    'average # of words']
    fio.WriteMatrix("../../data/TAC/statistics.txt", body, header)
    
def ExtractReferenceSummary(datadir, outdir, year, split=False):
    filelist = datadir+'list.json'
    data = fio.LoadDictJson(filelist)
    
    folder = year
    
    for lec, doc_id in enumerate(data[folder]):
        if split:
            if lec < len(data[folder])/2:
                suboutdir = outdir.replace(year, year+'_A')
            else:
                suboutdir = outdir.replace(year, year+'_B')
                lec = lec/2
        else:
            suboutdir = outdir
            
        path = os.path.join(suboutdir, str(lec))
        
        fio.NewPath(path)
        
        models = data[folder][doc_id]['models']
        
        assert(len(models) == 4)
        
        type= 'q1'
        
        for i, doc in enumerate(models):
            sentences = fio.LoadList(doc)
            filename = os.path.join(path, type+ '.ref.%d'%(i))
            print filename
            
            fio.SaveList(sentences, filename)
            
if __name__ == '__main__':
    datadir = "../../data/TAC/"    
    
#     ExtractDataStatitics(datadir, outdir)
    
    for year in ['s08', 's09', 's10', 's11',]:
        outdirs = ['../../data/TAC_%s/MC/'%(year),
                   '../../data/TAC_%s/ILP_MC/'%(year),
                   ]
        
        for outdir in outdirs:
            print outdir
            split = True
            
            ExtractSentence(datadir, outdir, year, split)
            ExtractReferenceSummary(datadir, outdir, year, split)
    
    #ss = ParseTACXML('../../data/TAC/s10\\test_doc_files\\D1035G\\D1035G-A\\new\\APW19980718.0011')
   