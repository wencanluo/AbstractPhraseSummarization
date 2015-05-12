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
        
def iter_documents(outdir, types, sheets = range(0,25), np='syntax', ngrams=[1]):
    """
    Generator: iterate over all relevant documents, yielding one
    document (=list of utf8 tokens) at a time.
    """
    print "types:", types
            
    # find all .txt documents, no matter how deep under top_directory
    for i, sheet in enumerate(sheets):
        week = i + 1
        dir = outdir + str(week) + '/'
        
        for question in types:
            prefix = dir + question + "." + np
            
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

def SaveNewA(A, dict, path, ngrams, prefixname=""):
    types = ['POI', 'MP', 'LP']
    
    sheets = range(0,25)
    
    TotoalLine = 0
    
    nBegin = 0
    
    for i in sheets:
        week = i + 1
        dir = path + str(week) + '/'
        
        for type in types:
            prefix = dir + type + "." + 'sentence'
            
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
                    
def getSVD(prefix, np, K, corpusname="corpus", ngrams=[1,2]): 
    types = ['POI', 'MP', 'LP']
    
    sheets = range(0,25)
            
    # that's it! the streamed corpus of sparse vectors is ready
    if corpusname=='book':
        corpus = BookCorpus(np, ngrams)
    else:
        corpus = TxtSubdirsCorpus(prefix, types, sheets, np, ngrams)
    
    path = prefix
    dictname = path + "_".join(types) + '_' + str(K) + '_' + corpusname + corpusdictexe
    fio.SaveDict(corpus.dictionary.token2id, dictname)

    # or run truncated Singular Value Decomposition (SVD) on the streamed corpus
    #from gensim.models.lsimodel import stochastic_svd as svd
    #u, s = svd(corpus, rank=300, num_terms=len(corpus.dictionary), chunksize=5000)
    
    #https://pypi.python.org/pypi/sparsesvd/
    scipy_csc_matrix = gensim.matutils.corpus2csc(corpus)
    print scipy_csc_matrix.shape
    
    SaveNewA(scipy_csc_matrix.toarray(), corpus.dictionary.token2id, path, ngrams, 'org')
    
    #rank = min(scipy_csc_matrix.shape) - 1
    rank = 200
    print rank
    
    #for L in [0.1, 0.5, 1, 0.05, 0.01, 0.005, 0.001]:
    for L in [2]:
        newA = softImputeWrapper.SoftImpute(scipy_csc_matrix.toarray().T, rank=rank, Lambda=L)
        
        print newA.shape
         
        newAname = path + "_".join(types) + '_' + str(K) + '_' + corpusname + mcexe
         
        prefix = str(rank) + '_' +  str(L)
        SaveNewA(newA, corpus.dictionary.token2id, path, ngrams, prefix)

def TestProcessLine():
    line = "how to determine the answers to part iii , in the activity ."
    print ProcessLine(line, [1, 2]).split()
    
    tokens = line.lower().split()
    
    ngrams = []
    for n in [1,2]:
        grams = ILP.getNgramTokenized(tokens, n, NoStopWords=True, Stemmed=True)
        ngrams = ngrams + grams
    print ngrams
                
if __name__ == '__main__':
    
    excelfile = "../../data/2011Spring_norm.xls"
    sennadatadir = "../../data/senna/"
    outdir = "../../data/SVD_Sentence/"
    bookname = "../../tools/TextBook_Materials.txt"
    
    #TestProcessLine()
    
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
    
    #SaveNewA(None, None, outdir)
    
    for np in ['sentence']:
        for K in [50]:
            getSVD(outdir, np, K, ngrams=[1,2])
            #getSVD(outdir, bookname, K, 'book', ngrams=[1,2])
    
    print "done"