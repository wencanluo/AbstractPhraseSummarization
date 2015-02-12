import json
import logging
import fio

phraseext = ".key" #a list
studentext = ".keys.source" #json
countext = ".dict"  #a dictionary
lpext = ".lp"
lpsolext = ".sol"
ngramext = ".ngram.json"
vectorext = ".ngram.vector.json"             

def CountWordVector(outdir, np):
    CountVector = {'False':{}, 'True':{}}
    
    sheets = range(0,12)
    for i, sheet in enumerate(sheets):
        week = i + 1
        dir = outdir + str(week) + '/'
        
        for type in ['POI', 'MP', 'LP']:
            prefix = dir + type + "." + np
            
            with open(prefix + vectorext, 'r') as infile:
                wordvector = json.load(infile)
            
            for k, v in wordvector.items():
                
                n = str(len(k.split()))
                if len(v) == 0:
                    key = 'False'
                else:
                    key = 'True'
                
                if n not in CountVector[key]:
                    CountVector[key][n] = {}
                
                if k not in CountVector[key][n]:
                    CountVector[key][n][k] = 0
                
                CountVector[key][n][k] = CountVector[key][n][k] + 1
    
    for n in ['1', '2']:
        for key in ['True', 'False']:
            fio.SaveDict(CountVector[key][n], outdir + "vector_"+key+"_"+n+".txt")
            
if __name__ == '__main__':
    
    excelfile = "../../data/2011Spring_norm.xls"
    sennadatadir = "../../data/senna/"
    outdir = "../../data/wordvector/"
    
    CountWordVector(outdir, np='syntax')
    
    print "done"