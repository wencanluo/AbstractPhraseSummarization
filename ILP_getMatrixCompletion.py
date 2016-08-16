#http://radimrehurek.com/2014/03/data-streaming-in-python-generators-iterators-iterables/
import json
import gensim
import fio
import numpy, scipy.sparse
from scipy.sparse.linalg import svds as sparsesvd
import re
import porter
import pickle
import softImputeWrapper
import ILP_baseline as ILP
import os
import global_params

phraseext = ".key" #a list
studentext = ".keys.source" #json
countext = ".dict"  #a dictionary
lpext = ".lp"
lpsolext = ".sol"
ngramext = ".ngram.json"
corpusdictexe = ".corpus.dict"
cscexe = ".mat.txt"
mcexe = ".mc.txt"

ngramTag = "___"

def ProcessLine(line,ngrams=[1]):
    #tokens = list(gensim.utils.tokenize(line, lower=True, errors='ignore'))
    tokens = line.lower().split()
    
    new_tokens = []
    for n in ngrams:
        ngram = ILP.getNgramTokenized(tokens, n, NoStopWords=True, Stemmed=True, ngramTag=ngramTag)
        new_tokens = new_tokens + ngram
    
    return " ".join(new_tokens)

def iter_folder(folder, extension, ngrams=[1]):
    for subdir, dirs, files in os.walk(folder):
        for file in sorted(files):
            if not file.endswith(extension): continue
            
            print file
            
            document = open(file).readlines()
            
            for line in document:
                line = ProcessLine(line, ngrams)
                #print line
                
                # break document into utf8 tokens
                yield gensim.utils.tokenize(line, lower=True, errors='ignore')    
    
def iter_documents(outdir, types, sheets = range(0,25), np='syntax', ngrams=[1]):
    """
    Generator: iterate over all relevant documents, yielding one
    document (=list of utf8 tokens) at a time.
    """
    print "types:", types
            
    # find all .txt documents, no matter how deep under top_directory
    for sheet in sheets:
        week = sheet
        dir = outdir + str(week) + '/'
        
        for question in types:
            prefix = dir + question + "." + np
            
            print prefix
            
            filename = prefix + phraseext
            if not fio.IsExist(filename): continue
            
            document = open(prefix + phraseext).readlines()
            
            for line in document:
                line = ProcessLine(line,ngrams)
                #print line
                
                # break document into utf8 tokens
                yield gensim.utils.tokenize(line, lower=True, errors='ignore')

def readbook(path, ngrams=[1]):
    document = open(path).readlines()
    
    for line in document:
        line = re.sub( '\s+', ' ', line).strip()
        if len(line) == 0: continue
        
        line = ProcessLine(line, ngrams)
        
        # break document into utf8 tokens
        yield gensim.utils.tokenize(line, lower=True, errors='ignore')
     
class TxtSubdirsCorpus(object):
    """
    Iterable: on each iteration, return bag-of-words vectors,
    one vector for each document.
 
    Process one document at a time using generators, never
    load the entire corpus into RAM.
 
    """
    def __init__(self, top_dir, types=['POI', 'MP', 'LP'], sheets = range(0,25), np='syntax', ngrams=[1]):
        self.types = types
        self.top_dir = top_dir
        self.np = np
        self.ngrams = ngrams
        self.sheets = sheets
        # create dictionary = mapping for documents => sparse vectors
        self.dictionary = gensim.corpora.Dictionary(iter_documents(top_dir, types, sheets, np, ngrams))
 
    def __iter__(self):
        """
        Again, __iter__ is a generator => TxtSubdirsCorpus is a streamed iterable.
        """
        for tokens in iter_documents(self.top_dir, self.types, self.sheets, self.np, self.ngrams):
            # transform tokens (strings) into a sparse vector, one at a time
            yield self.dictionary.doc2bow(tokens)

class TacCorpus(object):
    def __init__(self, top_dir, ngrams=[1]):
        self.top_dir = top_dir
        self.dictionary = gensim.corpora.Dictionary(iter_documents(top_dir, ngrams))
 
    def __iter__(self):
        """
        Again, __iter__ is a generator => TxtSubdirsCorpus is a streamed iterable.
        """
        for tokens in iter_folder(self.top_dir, self.ngrams):
            # transform tokens (strings) into a sparse vector, one at a time
            yield self.dictionary.doc2bow(tokens)

class BookCorpus(object):
    """
    Iterable: on each iteration, return bag-of-words vectors,
    one vector for each document.
 
    Process one document at a time using generators, never
    load the entire corpus into RAM.
 
    """
    def __init__(self, path, ngrams=[1]):
        self.path = path
        self.ngrams = ngrams
        # create dictionary = mapping for documents => sparse vectors
        self.dictionary = gensim.corpora.Dictionary(readbook(path, ngrams))
 
    def __iter__(self):
        """
        Again, __iter__ is a generator => TxtSubdirsCorpus is a streamed iterable.
        """
        for tokens in readbook(self.path, self.ngrams):
            # transform tokens (strings) into a sparse vector, one at a time
            yield self.dictionary.doc2bow(tokens)

def SaveCSC2(csc, filename):
    s = csc.shape
    m = s[0]
    n = s[1]
    
    body = []
    for i in range(m):
        row = []
        for j in range(n):
            row.append(csc[i, j])
        body.append(row)
    fio.WriteMatrix(filename, body, header=None)
    
def SaveCSC(csc, filename):
    A = csc.toarray()
    
    s = csc.shape
    m = s[0]
    n = s[1]
    
    data = []
    for i in range(m):
        row = []
        for j in range(n):
            x = A[i][j]
            if x != 0:
                row.append([j, A[i][j]])
        data.append(row)
    
    with open(filename, 'w') as fin:
        json.dump(data, fin, indent = 2)

def SaveSparseMatrix(A, filename):
    m = len(A)
    n = len(A[0])
    
    data = []
    for i in range(m):
        row = []
        for j in range(n):
            x = A[i][j]
            if x != 0:
                row.append([j, A[i][j]])
        data.append(row)
    
    with open(filename, 'w') as fin:
        json.dump(data, fin, indent = 2)

def SaveNewA(A, dict, path, ngrams, prefixname="", sheets = range(0,25), np='sentence', types=['POI', 'MP', 'LP']):
    TotoalLine = 0
        
    for i in sheets:
        week = i + 1
        dir = path + str(week) + '/'
        
        for type in types:
            prefix = dir + type + "." + np
            print prefix
            
            if not fio.IsExist(prefix + phraseext):
                print prefix + phraseext
                continue
            
            document = open(prefix + phraseext).readlines()
            
            LineRange = range(TotoalLine, TotoalLine + len(document))
            
            TotoalLine = TotoalLine + len(document)
            
            Bigrams = []
            for line in document:
                line = ProcessLine(line, ngrams)
                
                tokens = list(gensim.utils.tokenize(line, lower=True, errors='ignore'))
                
                Bigrams = Bigrams + tokens
            
            PartA = {}
            for bigram in set(Bigrams):
                if bigram not in dict:
                    print "error", bigram
                
                id = dict[bigram]
                
                row = A[id]
                
                PartA[bigram] = [row[x] for x in LineRange]
            
            svdAname = dir + type + '.' +prefixname + '.softA'
            print svdAname
            
            with open(svdAname, 'w') as fout:
                json.dump(PartA, fout, indent=2)            

def SaveNewA_New(A, dict, path, ngrams, prefixname="", sheets = range(0,25), np='sentence', types=['POI', 'MP', 'LP']):
    TotoalLine = 0
        
    for i in sheets:
        week = i
        dir = path + str(week) + '/'
        
        for type in types:
            prefix = dir + type + "." + np
            print prefix
            
            if not fio.IsExist(prefix + phraseext):
                print prefix + phraseext
                continue
            
            document = open(prefix + phraseext).readlines()
            
            LineRange = range(TotoalLine, TotoalLine + len(document))
            
            TotoalLine = TotoalLine + len(document)
            
            Bigrams = []
            for line in document:
                line = ProcessLine(line, ngrams)
                
                tokens = list(gensim.utils.tokenize(line, lower=True, errors='ignore'))
                
                Bigrams = Bigrams + tokens
            
            PartA = {}
            for bigram in set(Bigrams):
                if bigram not in dict:
                    print "error", bigram
                
                id = dict[bigram]
                
                row = A[id]
                
                PartA[bigram] = [row[x] for x in LineRange]
            
            svdAname = dir + type + '.' +prefixname + '.softA'
            print svdAname
            
            with open(svdAname, 'w') as fout:
                json.dump(PartA, fout, indent=2)        
                
def ToBinary(csc):
    A = csc.toarray()
    
    s = csc.shape
    m = s[0]
    n = s[1]
    
    m = len(A)
    n = len(A[0])
    
    for i in range(m):
        row = []
        for j in range(n):
            if A[i][j] >= 1: 
                A[i][j] = 1
    
    return A

def CheckBinary(A):
    m = len(A)
    n = len(A[0])
    
    for i in range(m):
        row = []
        for j in range(n):
            if A[i][j] != 0 and A[i][j] != 1: return False 
    
    return True   
                        
def getSVD(prefix, np, corpusname, ngrams, rank_max, softImpute_lambda, binary_matrix, output, types = ['POI', 'MP', 'LP']): 
    #types = ['POI', 'MP', 'LP']
    
    path = prefix
    
    sheets = range(0,26)
    dictname = output + "_".join(types) + '_' + corpusname + corpusdictexe
    
#     # that's it! the streamed corpus of sparse vectors is ready
#     if corpusname=='book':
#         corpus = BookCorpus(np, ngrams)
#     elif corpusname == 'tac':
#         corpus = TacCorpus(prefix, ngrams)
#         dictname = path + '_' + corpusname + corpusdictexe
#     else:
#         corpus = TxtSubdirsCorpus(prefix, types, sheets, np, ngrams)
#       
#     fio.SaveDict2Json(corpus.dictionary.token2id, dictname)
# 
#     # or run truncated Singular Value Decomposition (SVD) on the streamed corpus
#     #from gensim.models.lsimodel import stochastic_svd as svd
#     #u, s = svd(corpus, rank=300, num_terms=len(corpus.dictionary), chunksize=5000)
#      
#     #https://pypi.python.org/pypi/sparsesvd/
#     scipy_csc_matrix = gensim.matutils.corpus2csc(corpus)
#     print scipy_csc_matrix.shape
#      
#     print "binary_matrix: ", binary_matrix
#      
#     A = ToBinary(scipy_csc_matrix)
#      
#     rank = rank_max
#     print rank
#      
#     name = 'X'
#     newA = softImputeWrapper.SoftImpute(A.T, rank=rank, Lambda=softImpute_lambda, name=name, folder=output)
     
#     prefix = "org"
#     newA = softImputeWrapper.LoadA(name='X', folder=output)
    
    prefix = str(softImpute_lambda)
    Lambda = str(rank_max) + '_' + str(softImpute_lambda)
    newA = softImputeWrapper.LoadMC(Lambda=Lambda, name='newX', folder=output)
     
    if newA != None:
        print newA.shape
        
        token2id = fio.LoadDictJson(dictname)
        SaveNewA(newA, token2id, path, ngrams, prefix, np=np, types=types)

def getSVD_WriteX(cid, prefix, np, corpusname, ngrams, binary_matrix, output, types = ['POI', 'MP', 'LP']):
    path = prefix
    fio.NewPath(path)
    
    sheets = global_params.lectures[cid]
    dictname = output + "_".join(types) + '_' + corpusname + corpusdictexe
    
    # that's it! the streamed corpus of sparse vectors is ready
    if corpusname=='book':
        corpus = BookCorpus(np, ngrams)
    elif corpusname == 'tac':
        corpus = TacCorpus(prefix, ngrams)
        dictname = path + '_' + corpusname + corpusdictexe
    else:
        corpus = TxtSubdirsCorpus(prefix, types, sheets, np, ngrams)
       
    fio.SaveDict2Json(corpus.dictionary.token2id, dictname)
 
    #https://pypi.python.org/pypi/sparsesvd/
    scipy_csc_matrix = gensim.matutils.corpus2csc(corpus)
    print scipy_csc_matrix.shape
      
    print "binary_matrix: ", binary_matrix
      
    A = ToBinary(scipy_csc_matrix)
      
    name = 'X'
    softImputeWrapper.SoftImpute_SaveX(A.T, name=name, folder=output)

def getSVD_SaveOrg(cid, prefix, np, corpusname, ngrams, binary_matrix, output, types = ['POI', 'MP', 'LP']):
    path = prefix
    sheets = global_params.lectures[cid]
    dictname = output + "_".join(types) + '_' + corpusname + corpusdictexe
    
    prefix = "org"
    newA = softImputeWrapper.LoadA(name='X', folder=output)
      
    if newA != None:
        print newA.shape
         
        token2id = fio.LoadDictJson(dictname)
        SaveNewA_New(newA, token2id, path, ngrams, prefix, sheets=sheets, np=np, types=types)
    
def getSVD_LoadMC(cid, prefix, np, corpusname, ngrams, rank_max, softImpute_lambda, binary_matrix, output, types = ['POI', 'MP', 'LP']): 
    #types = ['POI', 'MP', 'LP']
    path = prefix
    sheets = global_params.lectures[cid]
    dictname = output + "_".join(types) + '_' + corpusname + corpusdictexe
        
    prefix = str(softImpute_lambda)
    Lambda = str(rank_max) + '_' + str(softImpute_lambda)
    newA = softImputeWrapper.LoadMC(Lambda=Lambda, name='newX', folder=output)
      
    if newA != None:
        print newA.shape
         
        token2id = fio.LoadDictJson(dictname)
        SaveNewA_New(newA, token2id, path, ngrams, prefix, sheets, np=np, types=types) 

def TestProcessLine():
    line = "how to determine the answers to part iii , in the activity ."
    print ProcessLine(line, [1, 2]).split()
    
    tokens = line.lower().split()
    
    ngrams = []
    for n in [1,2]:
        grams = ILP.getNgramTokenized(tokens, n, NoStopWords=True, Stemmed=True)
        ngrams = ngrams + grams
    print ngrams

             
def getMC_IE256():
    ILP_dir = "../../data/IE256/MC/" 
    outdir = "../../data/matrix/exp8/"
    
    #TestProcessLine()
    from config import ConfigFile
    
    config = ConfigFile(config_file_name='config_IE256.txt')
    
    for np in ['sentence']:
        
        getSVD_WriteX(ILP_dir, np, corpusname='corpus', ngrams=config.get_ngrams(), binary_matrix = config.get_binary_matrix(), output=outdir, types=['q1','q2'])
        
        for softImpute_lambda in numpy.arange(5.0, 5.1, 0.1):
            if softImpute_lambda < 1.4:
                rank_max = 2000
            else:
                rank_max = 500
            
            softImpute_lambda = "%.1f"%softImpute_lambda
            
            getSVD(ILP_dir, np, corpusname='corpus', ngrams=config.get_ngrams(), rank_max = rank_max, softImpute_lambda = softImpute_lambda, binary_matrix = config.get_binary_matrix(), output=outdir, types=['q1','q2'])

    print "done"
    
def getMC(cid):
    ILP_dir = "../../data/%s/MC/"%cid 
    outdir = "../../data/%s/matrix/exp5/"%cid
    fio.NewPath(outdir)
    
    #TestProcessLine()
    from config import ConfigFile
    
    config = ConfigFile(config_file_name='config_%s.txt'%cid)
    
    for np in ['sentence']:
#         getSVD_WriteX(cid, ILP_dir, np, corpusname='corpus', ngrams=config.get_ngrams(), binary_matrix = config.get_binary_matrix(), output=outdir, types=['q1','q2'])
        
#         getSVD_SaveOrg(cid, ILP_dir, np, corpusname='corpus', ngrams=config.get_ngrams(), binary_matrix = config.get_binary_matrix(), output=outdir, types=['q1','q2'])
        
        #pause, run the MC script
        
        for softImpute_lambda in numpy.arange(0.1, 8.0, 0.1):
            if softImpute_lambda < 1.4:
                rank_max = 2000
            else:
                rank_max = 500
             
            softImpute_lambda = "%.1f"%softImpute_lambda
                           
            getSVD_LoadMC(cid, ILP_dir, np, corpusname='corpus', ngrams=config.get_ngrams(), rank_max = rank_max, softImpute_lambda = softImpute_lambda, binary_matrix = config.get_binary_matrix(), output=outdir, types=['q1','q2'])

    print "done"
                        
if __name__ == '__main__':
    #getMC_IE256()
    getMC('IE256')
    #getMC('IE256_2016')
    #getMC('CS0445')
    exit(-1)
    
    excelfile = "../../data/2011Spring_norm.xls"
    sennadatadir = "../../data/senna/"
    
    ILP_dir = "../../data/IE256/MC/" 
    outdir = ILP_dir
    
    #TestProcessLine()
    from config import ConfigFile
    
    config = ConfigFile(config_file_name='tac_config.txt')
    
    for np in ['sentence']:
        for softImpute_lambda in numpy.arange(0.1, 8.0, 0.1):
            if softImpute_lambda < 1.4:
                rank_max = 2000
            else:
                rank_max = 500
            
            softImpute_lambda = "%.1f"%softImpute_lambda
            
            if softImpute_lambda.endswith('.0'):
                softImpute_lambda = softImpute_lambda[:-2]
            
            getSVD(ILP_dir, np, corpusname='corpus', ngrams=config.get_ngrams(), rank_max = rank_max, softImpute_lambda = softImpute_lambda, binary_matrix = config.get_binary_matrix(), output=outdir)

    print "done"