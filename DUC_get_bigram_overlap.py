import os
import TAC_ILP_baseline
import fio
import json
import numpy as np
import NLTKWrapper
import postProcess

def get_bigram_overlap(document, summary_file):
    sentences = [line.strip() for line in fio.ReadFile(document)]
    
    summaryList = [line.strip() for line in fio.ReadFile(summary_file)]
    
    bigram_ta = 0.
    covered_bigram_ta = 0.
    for summary in summaryList:
        bigrams = NLTKWrapper.getNgram(summary, 2)
        bigram_ta += len(bigrams)
        
        if bigram_ta == 0:
            print summary_file
            
        for token in bigrams:
            if postProcess.CheckKeyword(token, summaryList):
                covered_bigram_ta += 1
    
    if bigram_ta == 0:
        return 0.0
    
    return covered_bigram_ta/bigram_ta

def get_bigram_overlap_doc(prefix):
    document = prefix + '.key'
    
    rs = []
    summries = []
    for porfix in ['.ref1.summary', '.ref2.summary', '.ref3.summary', '.ref4.summary']:
        summary_file = prefix + porfix
        
        r = get_bigram_overlap(document, summary_file)
        
        rs.append(r)
    
    return np.average(rs)

def get_overlap():
    ilpdir = "../../data/DUC_ILP_Sentence/"
    
    data = {}
    for year in [
                 'DUC_2004', 
                 ]:
    
        path = os.path.join(ilpdir, year)
        print path
        for subdir, file in TAC_ILP_baseline.iter_folder(path, '.key'):
            doc_id = file[:-4]
            prefix = os.path.join(path, doc_id)
            print prefix
            print doc_id
            
            overlap = get_bigram_overlap_doc(prefix)
            
            data[doc_id] = overlap
            
    fio.SaveDict2Json(data, '../../data/DUC04/bigram_overlap.json')

def rank_rouge_by_overlap():
    bigram_overlap_json = '../../data/TAC/bigram_overlap.json'
    
    rouge_ILP_json = "../../data/TAC_ILP_Sentence/rouge.json"
    rouge_MC_json = "../../data/TAC_ILP_Sentence_MC/rouge.json"
    
    bigram_overlap = fio.LoadDictJson(bigram_overlap_json)
    rouge_ILP = fio.LoadDictJson(rouge_ILP_json)
    rouge_MC = fio.LoadDictJson(rouge_MC_json)
    
    doc_ids = sorted(bigram_overlap, key=bigram_overlap.get)
    
    RougeHeader = ['R1-R', 'R1-P', 'R1-F', 'R2-R', 'R2-P', 'R2-F', 'RSU4-R', 'RSU4-P', 'RSU4-F',]
    
    head = ['doc_id', 'overlap'] + RougeHeader + RougeHeader
    
    body = []
    for doc_id in doc_ids:
        row = [doc_id]
        row.append(bigram_overlap[doc_id])
        row = row + rouge_ILP[doc_id]
        row = row + rouge_MC[doc_id]
        body.append(row)
    
    fio.WriteMatrix('../../data/TAC/rouge_sort.txt', body, head)
            
if __name__ == '__main__':
    #rank_rouge_by_overlap()
    
    #get_overlap_IE256()
    
    get_overlap()