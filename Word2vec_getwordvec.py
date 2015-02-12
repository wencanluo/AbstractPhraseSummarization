import json
import logging

phraseext = ".key" #a list
studentext = ".keys.source" #json
countext = ".dict"  #a dictionary
lpext = ".lp"
lpsolext = ".sol"
ngramext = ".ngram.json"
vectorext = ".ngram.vector.json"

import gensim
model = gensim.models.word2vec.Word2Vec.load_word2vec_format('../../tools/GoogleNews-vectors-negative300.bin', binary=True)
print "load word2vec success"
#model = {'the':[1,1,0]}

def getWordVector(prefix):
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    
    #extract the ngram from the phrase
    ngramfile = prefix+ngramext
    
    with open(ngramfile, 'r') as infile:
        ngramdict = json.load(infile)
    
    wordvector = {}
    for n in ngramdict:
        ngrams = ngramdict[n]
        
        for ngram in ngrams:
            if int(n) > 1:
                phrase = ngram.replace(" ", "_")
            else:
                phrase = ngram
            
            if phrase not in model:
                wordvector[ngram] = []
            else:
                vec = model[phrase]
                wordvector[ngram] = [float(v) for v in vec]
    
    with open(prefix + vectorext, 'w') as outfile:
        json.dump(wordvector, outfile, indent=2)  
             

def ExtractWordVector(outdir, np):
    sheets = range(0,12)
    for i, sheet in enumerate(sheets):
        week = i + 1
        dir = outdir + str(week) + '/'
        
        for type in ['POI', 'MP', 'LP']:
            prefix = dir + type + "." + np
            
            getWordVector(prefix)
            
if __name__ == '__main__':
    
    excelfile = "../../data/2011Spring_norm.xls"
    sennadatadir = "../../data/senna/"
    outdir = "../../data/wordvector/"
    
    #Step1: get senna input
    #Survey.getStudentResponses4Senna(excelfile, sennadatadir)
    
    #Step2: get senna output
    
    #Step3: get phrases
    #for np in ['syntax', 'chunk']:
#     for np in ['syntax']:
#          postProcess.ExtractNPFromRaw(excelfile, sennadatadir, outdir, method=np)
#          postProcess.ExtractNPSource(excelfile, sennadatadir, outdir, method=np)
#          postProcess.ExtractNPFromRawWithCount(excelfile, sennadatadir, outdir, method=np)
#     
#     #Step4: write TA's reference 
#     Survey.WriteTASummary(excelfile, outdir)
    
    for np in ['syntax']:
        ExtractWordVector(outdir, np)
    
    print "done"