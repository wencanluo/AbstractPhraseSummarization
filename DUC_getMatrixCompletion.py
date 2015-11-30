#http://radimrehurek.com/2014/03/data-streaming-in-python-generators-iterators-iterables/
import json
import gensim
import fio
import numpy, scipy.sparse
import re
import porter
import pickle
import softImputeWrapper
import ILP_baseline as ILP
import os

phraseext = ".key" #a list
studentext = ".keys.source" #json
countext = ".dict"  #a dictionary
lpext = ".lp"
lpsolext = ".sol"
ngramext = ".ngram.json"
corpusdictexe = ".corpus.dict.json"
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

def iter_document(input, ngrams=[1]):
    document = open(input).readlines()
            
    for line in document:
        line = ProcessLine(line, ngrams)
        #print line
        
        # break document into utf8 tokens
        yield gensim.utils.tokenize(line, lower=True, errors='ignore')
    

def iter_folder(folder, extension, ngrams=[1]):
    for subdir, dirs, files in os.walk(folder):
        for file in sorted(files):
            if not file.endswith(extension): continue
            
            print file
            
            document = open(os.path.join(subdir, file)).readlines()
            
            for line in document:
                line = ProcessLine(line, ngrams)
                #print line
                
                # break document into utf8 tokens
                yield gensim.utils.tokenize(line, lower=True, errors='ignore')    
    
class TacCorpus(object):
    def __init__(self, top_dir, ngrams=[1]):
        self.top_dir = top_dir
        self.ngrams = ngrams
        self.dictionary = gensim.corpora.Dictionary(iter_folder(top_dir, '.key', ngrams))
 
    def __iter__(self):
        """
        Again, __iter__ is a generator => TxtSubdirsCorpus is a streamed iterable.
        """
        for tokens in iter_folder(self.top_dir, '.key', self.ngrams):
            # transform tokens (strings) into a sparse vector, one at a time
            yield self.dictionary.doc2bow(tokens)

class TacCorpusDocument(object):
    def __init__(self, top_dir, ngrams=[1]):
        self.top_dir = top_dir
        self.ngrams = ngrams
        self.dictionary = gensim.corpora.Dictionary(iter_document(top_dir, ngrams))
 
    def __iter__(self):
        """
        Again, __iter__ is a generator => TxtSubdirsCorpus is a streamed iterable.
        """
        for tokens in iter_document(self.top_dir, self.ngrams):
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
    prefix = path
    print prefix
    
    document = open(prefix + phraseext).readlines()
    
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

        PartA[bigram] = [x for x in A[id]]
    
    svdAname = prefix + '.'+ prefixname + '.softA'
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
                        
def getSVD(prefix, np, corpusname, ngrams, rank_max, softImpute_lambda, binary_matrix): 
    path = prefix
    name = prefix[prefix.rfind('/')+1:]
    folder = prefix[:prefix.rfind('/')]
    
    dictname = path + corpusdictexe
    
    # that's it! the streamed corpus of sparse vectors is ready
    corpus = TacCorpusDocument(prefix + '.key', ngrams)
     
    fio.SaveDict2Json(corpus.dictionary.token2id, dictname)
    #fio.SaveDict(corpus.dictionary.token2id, dictname)
 
    # or run truncated Singular Value Decomposition (SVD) on the streamed corpus
    #from gensim.models.lsimodel import stochastic_svd as svd
    #u, s = svd(corpus, rank=300, num_terms=len(corpus.dictionary), chunksize=5000)
     
    #https://pypi.python.org/pypi/sparsesvd/
    scipy_csc_matrix = gensim.matutils.corpus2csc(corpus)
    print scipy_csc_matrix.shape
     
    print "binary_matrix: ", binary_matrix
     
    if binary_matrix:
        A = ToBinary(scipy_csc_matrix)
        #SaveNewA(A, corpus.dictionary.token2id, path, ngrams, 'org.binary')
    else:
        A = scipy_csc_matrix.toarray()
        #SaveNewA(scipy_csc_matrix.toarray(), corpus.dictionary.token2id, path, ngrams, 'org')
     
    rank = rank_max
    print rank
       
    newA = softImputeWrapper.SoftImpute(A.T, rank=rank, Lambda=softImpute_lambda, name=name, folder=folder)
    
#     
#     prefix = str(softImpute_lambda)
#     
#     svdAname = path + '.'+ prefix + '.softA'
#     #if fio.IsExist(svdAname): return
#         
#     #newA = softImputeWrapper.LoadMC(Lambda=softImpute_lambda, name=name, folder=folder)
#     
#     #newA = softImputeWrapper.LoadA(name=name, folder=folder)
#     #prefix = 'org'
#     
#     if newA != None:
#         print newA.shape
#         
#         token2id = fio.LoadDictJson(dictname)
#         SaveNewA(newA, token2id, path, ngrams, prefix)

def TestProcessLine():
    line = "how to determine the answers to part iii , in the activity ."
    print ProcessLine(line, [1, 2]).split()
    
    tokens = line.lower().split()
    
    ngrams = []
    for n in [1,2]:
        grams = ILP.getNgramTokenized(tokens, n, NoStopWords=True, Stemmed=True)
        ngrams = ngrams + grams
    print ngrams

def generate_task(folder, year):
    import sys
    sys.stdout = open('run_duc_mc_%s.bat'%year, 'w')
    
    for subdir, dirs, files in os.walk(folder + year):
        for file in sorted(files):
            if not file.endswith('.key'): continue
            
            name = file[:-4]
            
            print 'rscript', "softImpute_DUC.R", year, name
    
    print "pause"
    
    sys.stdout.close()
    
if __name__ == '__main__':
    outdirs = [#'../../data/TAC_MC/s08/',
               #'../../data/TAC_MC/s09/',
               '../../data/DUC_MC/DUC_2004/',
               #'../../data/TAC_MC/s11/',
               ]
    from config import ConfigFile
    
    config = ConfigFile(config_file_name='duc_config.txt')
    
    np = 'sentence'
    
#     import sys
#     for year in ['DUC_2004']:
#         generate_task('../../data/DUC_MC/', year)
#     exit(-1)
     
    for m_lambda in ['5']:
        for folder in outdirs:
            for subdir, dirs, files in os.walk(folder):
                for file in sorted(files):
                    if not file.endswith('.key'): continue
                    
                    filename = os.path.join(subdir, file)
                    getSVD(filename[:-4], np, corpusname='tac', ngrams=config.get_ngrams(), rank_max = config.get_rank_max(), softImpute_lambda = m_lambda, binary_matrix = config.get_binary_matrix())
            
#     for np in ['sentence']:
#         getSVD(outdir, np, corpusname='tac', ngrams=config.get_ngrams(), rank_max = config.get_rank_max(), softImpute_lambda = config.get_softImpute_lambda(), binary_matrix = config.get_binary_matrix())
#     
    print "done"