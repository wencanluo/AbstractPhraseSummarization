import os, fio
import ILP_getMatrixCompletion as ILP_getMC
import ILP_MC
import numpy as np
import json
from scipy.stats import entropy

phraseext = ".key" #a list
studentext = ".keys.source" #json
countext = ".dict"  #a dictionary
orgA_ext = '.org.softA'

class MCCorpus:
    def __init__(self, folder):
        self.folder= os.path.join(folder, 'ILP_MC')
        self.mc_folder = os.path.join(folder, 'MC')
        
        self.maxLec = 100
        self.ngrams = [2] #bigram only
        self.task_prefix = []
        self.keyfiles = []
        self.bigrams = []
        self.As = []
        self.N = 2
        
    def load_task(self):
        for lec in range(self.maxLec):
            lec_folder = os.path.join(self.folder, str(lec))
            if not fio.IsExistPath(lec_folder): continue
            
            for prompt in ['q1', 'q2', 'q3', 'POI', 'MP', 'LP']:
                prefix = os.path.join(lec_folder, prompt + '.sentence')
                if not fio.IsExist(prefix + phraseext): continue
                
                self.task_prefix.append(prefix)
    
    def load_keyfiles(self):
        for prefix in self.task_prefix:
            key_file = prefix + phraseext
            
            doc = open(key_file).readlines()
            self.keyfiles.append(doc)
    
    def get_number_of_sentence(self):
        if not self.keyfiles:
            self.load_keyfiles()
        
        counts = [] 
        for doc in self.keyfiles:
            counts.append(len(doc))
        self.number_of_sentences = counts
        self.total_number_of_sentences = sum(self.number_of_sentences)
        
        return self.total_number_of_sentences
    
    def extract_bigrams(self):
        for doc in self.keyfiles:
            bigrams = []
            for line in doc:
                bigram_tokens = ILP_getMC.ProcessLine(line, self.ngrams)
                bigrams.append(len(bigram_tokens.split()))
            self.bigrams.append(bigrams)
    
    def get_number_of_sentence_per_task(self):
        return '%.1f'%np.mean(self.number_of_sentences)

    def get_task_num(self):
        self.task_num = len(self.task_prefix)
        return self.task_num
    
    def get_bigram_num(self):
        if not self.bigrams:
            self.extract_bigrams()
        self.bigram_num = np.sum(np.sum(self.bigrams)) 
        return self.bigram_num
    
    def get_bigram_num_per_task(self):
        return '%.1f'% (self.bigram_num/float(self.task_num))
    
    def get_bigram_num_per_sentence(self):
        return '%.1f'%(self.bigram_num/float(self.total_number_of_sentences))
    
    def get_size(self):
        return self.total_number_of_sentences * self.bigram_num
    
    def load_A(self):
        self.As = []
        
        for prefix in self.task_prefix:
            aprefix = prefix.replace('ILP_MC', 'MC').replace('.sentence', '')
            A_file = aprefix +  orgA_ext
            
            bigramDict = ILP_MC.LoadMC(A_file)
            self.As.append(bigramDict)
    
    def get_sparsity_ratio_A(self, A):
        zero, total = 0, 0.
        for bigram in A:
            total += len(A[bigram])
            for val in A[bigram]:
                if val == 0:
                    zero += 1
        return zero/total
    
    def get_bigram_ratio_N_A(self, A):
        total = float(len(A))
        count = 0
        for bigram in A:
            if sum(A[bigram]) >= self.N:
                count += 1
        return count/total
    
    def get_bigram_entropy_A(self, A):
        counts = []
        for bigram in A:
            counts.append(sum(A[bigram]))
        total = float(sum(counts))
        return entropy(np.array(counts)/total) 
    
    def get_sparsity_ratio(self):
        if not self.As:
            self.load_A()
            
        self.sparsities = []
        
        for A in self.As:
            self.sparsities.append(self.get_sparsity_ratio_A(A))
        
        return '%.3f%%'%(np.mean(self.sparsities)*100)
    
    def get_bigram_ratio_more_than_N(self):
        self.bigram_ratio_Ns = []
        
        for A in self.As:
            self.bigram_ratio_Ns.append(self.get_bigram_ratio_N_A(A))
        
        return '%.3f%%'%(np.mean(self.bigram_ratio_Ns)*100)
    
    def get_bigram_entropy(self):
        self.bigram_entropies = []
        
        for A in self.As:
            self.bigram_entropies.append(self.get_bigram_entropy_A(A))
        
        return '%.4f'%(np.mean(self.bigram_entropies))
        
    
def get_summary(datadir, corpus, output):
    
    head = ['name', 'domain', 'task #', 'sen #', 'sen #/task', 'bigram #', 'bigram #/task', 'bigram #/sentence', 'size of A', 
            'sparsity of A', 'ratio of bigram >=2', 'Shannon index'
            ]
    
    body = []
    for name, domain in corpus:
        row = [name, domain]
        folder = os.path.join(datadir, name)
        
        mc_corpus = MCCorpus(folder)
        mc_corpus.load_task()
        
        row.append(mc_corpus.get_task_num())
        row.append(mc_corpus.get_number_of_sentence())
        row.append(mc_corpus.get_number_of_sentence_per_task())
        row.append(mc_corpus.get_bigram_num())
        row.append(mc_corpus.get_bigram_num_per_task())
        row.append(mc_corpus.get_bigram_num_per_sentence())
        row.append(mc_corpus.get_size())
        
        row.append(mc_corpus.get_sparsity_ratio())
        row.append(mc_corpus.get_bigram_ratio_more_than_N())
        row.append(mc_corpus.get_bigram_entropy())
        
        body.append(row)
    
    fio.WriteMatrix(output, body, head)
        
if __name__ == '__main__':
    
    datadir = '../../data/'
    corpus = [
            ('Engineer','response'),
            ('IE256','response'),
            ('IE256_2016','response'),
            ('CS0445','response'),
            ('review_camera','review'),
            ('review_prHistory','review'),
            ('review_IMDB','review'),
#               ('DUC04', )
#               ('TAC08', )
#               ('TAC09', )
#               ('TAC10', )
#               ('TAC11', )
              ]
    
    output = '../../data/statistcis_all.txt'
    get_summary(datadir, corpus, output)