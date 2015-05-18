import softImputeWrapper
import SVD_getMatrixCompletion
import fio
import re
from config import dict_ngrams, dict_lambda
        
def extract_mc(path, ngrams, rank_max, softImpute_lambda): 
    newA = softImputeWrapper.LoadR(rank_max, softImpute_lambda, path=path)
    
    if newA != None:
        dict = fio.LoadDict(path + 'POI_MP_LP_corpus.corpus.dict', 'int')
        
        prefixname = str(rank_max) + '_' +  str(softImpute_lambda)
        SVD_getMatrixCompletion.SaveNewA(newA, dict, path, ngrams, prefixname, sheets = range(0,12))

def extract_orgA(path, ngrams): 
    newA = softImputeWrapper.LoadX(path=path)
    
    if newA != None:
        dict = fio.LoadDict(path + 'POI_MP_LP_corpus.corpus.dict', 'int')
        
        prefixname = "org"
        SVD_getMatrixCompletion.SaveNewA(newA, dict, path, ngrams, prefixname, sheets = range(0,12))
        
def get_lambda(logfile, rank):
    #get the best lambda for a rank
    
    lines = fio.ReadFile(logfile)
    
    for line in lines:
        g = re.match('\d+ lambda= ([0-9]*\.?[0-9]+) rank.max (\d+) rank (\d+) ', line)
        if g:
            print g.group(1), g.group(2), g.group(3)

def Write_Sentence(excelfile, sennadatadir, path):
    import postProcess
    postProcess.ExtractNPFromRaw(excelfile, sennadatadir, path, method='sentence', weekrange=range(0,12))
                     
if __name__ == '__main__':
    
    excelfile = "../../data/2011Spring_norm.xls"
    sennadatadir = "../../data/senna/"
    outdir = "../../data/SVD_Sentence/"
    
    #get_lambda('log4.txt', 20)
    
    for exp in [1, 2, 3, 4]:
        path = 'E:/project/AbstractPhraseSummarization/data/matrix/exp' + str(exp) + '/'
        #Write_Sentence(excelfile, sennadatadir, path)
        ngram = dict_ngrams[exp]
        extract_orgA(path, ngrams=ngram)
        
        for rank in [20, 50, 100, 200, 500, 1000, 2690]:
            Lambda = dict_lambda[exp][rank]
            #for Lambda in [10000, 100, 10, 8, 4, 3, 2.5, 2, 1.5, 1, 0.5, 0.1, 0.01]:
            extract_mc(path, ngrams=ngram, rank_max=rank, softImpute_lambda=Lambda)
        
    print "done"