import numpy
import scipy.sparse
import gensim
import SVD_getUS2
import fio

from scipy.sparse.linalg import svds as sparsesvd

def ProcessLine(line,ngrams=[1]):
    #tokens = list(gensim.utils.tokenize(line, lower=True, errors='ignore'))
    tokens = line.lower().split()
    
    new_tokens = []
    for n in ngrams:
        ngram = SVD_getUS2.getNgramTokenized(tokens, n, NoStopWords=True, Stemmed=True)
        new_tokens = new_tokens + ngram
    
    return " ".join(new_tokens)

def iter_documents():     
    document = open("../../data/svd_test.txt").readlines()
    
    for line in document:
        line = ProcessLine(line,ngrams=[1,2])
        #print line
        
        # break document into utf8 tokens
        yield gensim.utils.tokenize(line, lower=True, errors='ignore')

class TxtSubdirsCorpus(object):
    """
    Iterable: on each iteration, return bag-of-words vectors,
    one vector for each document.
 
    Process one document at a time using generators, never
    load the entire corpus into RAM.
 
    """
    def __init__(self):
        # create dictionary = mapping for documents => sparse vectors
        self.dictionary = gensim.corpora.Dictionary(iter_documents())
 
    def __iter__(self):
        """
        Again, __iter__ is a generator => TxtSubdirsCorpus is a streamed iterable.
        """
        for tokens in iter_documents():
            # transform tokens (strings) into a sparse vector, one at a time
            yield self.dictionary.doc2bow(tokens)
                            
def TestSimpleInput():
    corpus = TxtSubdirsCorpus()
    
    scipy_csc_matrix = gensim.matutils.corpus2csc(corpus)
    dictname = "../../data/svd_test_tokens.txt"
    
    fio.SaveDict(corpus.dictionary.token2id, dictname)
    
    PrintMatrix(scipy_csc_matrix.toarray())
    
    K = 3
    
    csc_m = scipy.sparse.csc_matrix(scipy_csc_matrix.T)
    u, s, vt = sparsesvd(csc_m, K)
    new_matrix = numpy.dot(u, numpy.dot(numpy.diag(s), vt))
        
    #print new_matrix
    PrintMatrix(new_matrix.T)
    
    K = 2
    
    csc_m = scipy.sparse.csc_matrix(scipy_csc_matrix.T)
    u, s, vt = sparsesvd(csc_m, K)
    new_matrix = numpy.dot(u, numpy.dot(numpy.diag(s), vt))
        
    #print new_matrix
    PrintMatrix(new_matrix.T)

def PrintMatrix(A):
    m,n = A.shape
    
    for i in range(m):
        for j in range(n):
            print '%.1f' % A[i][j],'\t',
        print
    print

def TestRandomMatrix():
    row, col = 15, 10
    K = 4
    
    ratio = 0.1
    matrix = numpy.random.choice([0.0, 1.0], size=(row, col), p=[1-ratio, ratio])
    #print matrix
    
    PrintMatrix(matrix)
    
    csc_m = scipy.sparse.csc_matrix(matrix)
    u, s, vt = sparsesvd(csc_m, K)
    new_matrix = numpy.dot(u, numpy.dot(numpy.diag(s), vt))
        
    #print new_matrix
    PrintMatrix(new_matrix)
    

def TestSVD():
    #TestRandomMatrix()
    
    TestSimpleInput()

if __name__ == '__main__':
    TestSVD()