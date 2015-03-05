#http://radimrehurek.com/2014/03/data-streaming-in-python-generators-iterators-iterables/
import json
import gensim
import fio
import numpy, scipy.sparse
from scipy.sparse.linalg import svds as sparsesvd
import re

phraseext = ".key" #a list
studentext = ".keys.source" #json
countext = ".dict"  #a dictionary
lpext = ".lp"
lpsolext = ".sol"
ngramext = ".ngram.json"
corpusdictexe = ".corpus.dict"
cscexe = ".mat.txt"
ngramTag = "___"

stopwords = [line.lower().strip() for line in fio.ReadFile("../../../Fall2014/summarization/ROUGE-1.5.5/data/smart_common_words.txt")]
punctuations = ['.', '?', '-', ',', '[', ']', '-', ';', '\'', '"', '+', '&', '!', '/', '>', '<', ')', '(', '#', '=']

stopwordswithpunctuations = stopwords + punctuations

def save_sparse_csr(filename, array):
    numpy.savez(filename, data = array.data, indices=array.indices,
             indptr = array.indptr, shape=array.shape)

def load_sparse_csr(filename):
    loader = np.load(filename)
    return csr_matrix((  loader['data'], loader['indices'], loader['indptr']),
                         shape = loader['shape'])

def getNgramTokenized(tokens, n, NoStopWords=False):
    #n is the number of grams, such as 1 means unigram
    ngrams = []
    
    N = len(tokens)
    for i in range(N):
        if i+n > N: continue
        ngram = tokens[i:i+n]
        
        if not NoStopWords:
            ngrams.append(ngramTag.join(ngram))
        else:
            removed = True
            for w in ngram:
                if w not in stopwords:
                    removed = False
            
            if not removed:
                ngrams.append(ngramTag.join(ngram))
            
    return ngrams

def removeStopWords(tokens):
    newTokens = [token for token in tokens if token.lower() not in stopwordswithpunctuations]
    return newTokens

def ProcessLine(line,ngrams=[1]):
    tokens = list(gensim.utils.tokenize(line, lower=True, errors='ignore'))
    
    #tokens = removeStopWords(tokens)
    
    new_tokens = []
    for n in ngrams:
        ngram = getNgramTokenized(tokens, n, NoStopWords=True)
        new_tokens = new_tokens + ngram
    
    return " ".join(new_tokens)
        
def iter_documents(outdir, types, np='syntax', ngrams=[1]):
    """
    Generator: iterate over all relevant documents, yielding one
    document (=list of utf8 tokens) at a time.
    """
    # find all .txt documents, no matter how deep under top_directory
    sheets = range(0,25)
    
    for i, sheet in enumerate(sheets):
        week = i + 1
        dir = outdir + str(week) + '/'
        
        for type in types:
            prefix = dir + type + "." + np
            
            document = open(prefix + phraseext).readlines()
            
            onedocument = []
            for line in document:
                line = ProcessLine(line,ngrams)
                onedocument.append(line)
                
            # break document into utf8 tokens
            yield gensim.utils.tokenize(" ".join(onedocument), lower=True, errors='ignore')

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
    def __init__(self, top_dir, types=['POI', 'MP', 'LP'], np='syntax', ngrams=[1]):
        self.types = types
        self.top_dir = top_dir
        self.np = np
        self.ngrams = ngrams
        # create dictionary = mapping for documents => sparse vectors
        self.dictionary = gensim.corpora.Dictionary(iter_documents(top_dir, types, np, ngrams))
 
    def __iter__(self):
        """
        Again, __iter__ is a generator => TxtSubdirsCorpus is a streamed iterable.
        """
        for tokens in iter_documents(self.top_dir, self.types, self.np, self.ngrams):
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

def WeightbyDF(csc):
    s = csc.shape
    m = s[0]
    n = s[1]
    
    firstrow = []
    for j in range(n):
        firstrow.append(csc[0,j])
    print firstrow
    
    for i in range(m):
        df = 0.0 #get document frequency
        for j in range(n):
            if csc[i, j] != 0:
                df = df + 1.0
        
        for j in range(n):
            if csc[i, j] != 0:
                csc[i, j] = csc[i, j] / df
    
    firstrow = []
    for j in range(n):
        firstrow.append(csc[0,j])
    print firstrow
    
    return csc
               
def getSVD(prefix, np, K, corpusname="corpus", ngrams=[1,2]): 
    types = ['POI', 'MP', 'LP']
    
    # that's it! the streamed corpus of sparse vectors is ready
    if corpusname=='book':
        corpus = BookCorpus(np, ngrams)
    else:
        corpus = TxtSubdirsCorpus(prefix,types, np, ngrams)
    dictname = prefix + "_".join(types) + '_' + str(K) + '_' + corpusname+corpusdictexe
    fio.SaveDict(corpus.dictionary.token2id, dictname)
     
    # or run truncated Singular Value Decomposition (SVD) on the streamed corpus
    #from gensim.models.lsimodel import stochastic_svd as svd
    #u, s = svd(corpus, rank=300, num_terms=len(corpus.dictionary), chunksize=5000)
    
    #https://pypi.python.org/pypi/sparsesvd/
    scipy_csc_matrix = gensim.matutils.corpus2csc(corpus)
    print scipy_csc_matrix.shape
    
    scipy_csc_matrix = WeightbyDF(scipy_csc_matrix)
    print scipy_csc_matrix.shape
    
#     for vector in scipy_csc_matrix:
#         print vector
    
    #http://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.sparse.linalg.svds.html
    u, s, vt = sparsesvd(scipy_csc_matrix, K) # do SVD, asking for 100 factors
     
    print type(u), u.shape
    #save_sparse_csr(prefix + "u" + cscexe, u)
    #u.tofile(prefix + "u" + cscexe)
    numpy.savetxt(prefix + "u" + '_'+ str(K) + '_' + corpusname + cscexe, u)
    
    print s.shape
    #save_sparse_csr(prefix + "s" + cscexe, s)
    #s.tofile(prefix + "s" + cscexe)
    numpy.savetxt(prefix + "s" + '_'+ str(K) + '_' + corpusname+cscexe, s)
    
    print vt.shape
    #save_sparse_csr(prefix + "vt" + cscexe, vt)
    #vt.tofile(prefix + "vt" + cscexe)
    numpy.savetxt(prefix + "vt" + '_'+ str(K) + '_' + corpusname+cscexe, vt)
    
if __name__ == '__main__':
    
    excelfile = "../../data/2011Spring_norm.xls"
    sennadatadir = "../../data/senna/"
    outdir = "../../data/SVD_Sentence_Lecture/"
    bookname = "../../tools/TextBook_Materials.txt"
    
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
    
    for np in ['sentence']:
        #for K in [50, 100, 200]:
        for K in [50]:
            getSVD(outdir, np, K, ngrams=[1,2])
            #getSVD(outdir, bookname, K, 'book', ngrams=[1,2])
    
    print "done"