#http://radimrehurek.com/2014/03/data-streaming-in-python-generators-iterators-iterables/
import json
import gensim
import fio
import numpy, scipy.sparse
from scipy.sparse.linalg import svds as sparsesvd

phraseext = ".key" #a list
studentext = ".keys.source" #json
countext = ".dict"  #a dictionary
lpext = ".lp"
lpsolext = ".sol"
ngramext = ".ngram.json"
corpusdictexe = ".corpus.dict"
cscexe = ".mat.txt"

def save_sparse_csr(filename, array):
    numpy.savez(filename, data = array.data, indices=array.indices,
             indptr = array.indptr, shape=array.shape)

def load_sparse_csr(filename):
    loader = np.load(filename)
    return csr_matrix((  loader['data'], loader['indices'], loader['indptr']),
                         shape = loader['shape'])
    
def iter_documents(outdir, types, np='syntax'):
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
            
            for line in document:
                # break document into utf8 tokens
                yield gensim.utils.tokenize(line, lower=True, errors='ignore')
 
class TxtSubdirsCorpus(object):
    """
    Iterable: on each iteration, return bag-of-words vectors,
    one vector for each document.
 
    Process one document at a time using generators, never
    load the entire corpus into RAM.
 
    """
    def __init__(self, top_dir, types=['POI', 'MP', 'LP'], np='syntax'):
        self.types = types
        self.top_dir = top_dir
        self.np = np
        # create dictionary = mapping for documents => sparse vectors
        self.dictionary = gensim.corpora.Dictionary(iter_documents(top_dir, types, np))
 
    def __iter__(self):
        """
        Again, __iter__ is a generator => TxtSubdirsCorpus is a streamed iterable.
        """
        for tokens in iter_documents(self.top_dir, self.types, self.np):
            # transform tokens (strings) into a sparse vector, one at a time
            yield self.dictionary.doc2bow(tokens)

def getSVD(prefix): 
    types = ['POI', 'MP', 'LP']
    
    # that's it! the streamed corpus of sparse vectors is ready
    corpus = TxtSubdirsCorpus(prefix,types, np='syntax')
    dictname = prefix+"_".join(types)+corpusdictexe
    fio.SaveDict(corpus.dictionary.token2id, dictname)
     
    # or run truncated Singular Value Decomposition (SVD) on the streamed corpus
    #from gensim.models.lsimodel import stochastic_svd as svd
    #u, s = svd(corpus, rank=300, num_terms=len(corpus.dictionary), chunksize=5000)
    
    #https://pypi.python.org/pypi/sparsesvd/
    scipy_csc_matrix = gensim.matutils.corpus2csc(corpus)
    print scipy_csc_matrix.shape
    
#     for vector in scipy_csc_matrix:
#         print vector
    
    #http://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.sparse.linalg.svds.html
    u, s, vt = sparsesvd(scipy_csc_matrix, 300) # do SVD, asking for 100 factors
     
    print type(u), u.shape
    #save_sparse_csr(prefix + "u" + cscexe, u)
    #u.tofile(prefix + "u" + cscexe)
    numpy.savetxt(prefix + "u" + cscexe, u)
    
    print s.shape
    #save_sparse_csr(prefix + "s" + cscexe, s)
    #s.tofile(prefix + "s" + cscexe)
    numpy.savetxt(prefix + "s" + cscexe, s)
    
    print vt.shape
    #save_sparse_csr(prefix + "vt" + cscexe, vt)
    #vt.tofile(prefix + "vt" + cscexe)
    numpy.savetxt(prefix + "vt" + cscexe, vt)
    
if __name__ == '__main__':
    
    excelfile = "../../data/2011Spring_norm.xls"
    sennadatadir = "../../data/senna/"
    outdir = "../../data/SVD/"
    
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
        getSVD(outdir)
    
    print "done"