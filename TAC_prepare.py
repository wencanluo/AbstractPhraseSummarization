import fio
import os
import json
import re
import xml.etree.ElementTree as ET
import NLTKWrapper

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
                
            
if __name__ == '__main__':
    datadir = "../../data/TAC/"
    
    outdir  = "../../data/TAC_ILP/"
    
    ExtractSentence(datadir, outdir)
    
    #ss = ParseTACXML('../../data/TAC/s10\\test_doc_files\\D1035G\\D1035G-A\\new\\APW19980718.0011')
   