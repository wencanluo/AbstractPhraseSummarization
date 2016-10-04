import os, fio
import ILP_getMatrixCompletion as ILP_getMC
import ILP_MC
import numpy as np
import json
from scipy.stats import entropy
import collections

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
        self.summary_files = []
        
        self.keyfiles = []
        self.sumfiles = []
        
        self.bigrams = []
        self.bigram_counts = []
        self.As = []
        self.N = 2 #bigram cut
        
        self.sum_bigram_counts = []
        
    def load_task(self):
        for lec in range(self.maxLec):
            lec_folder = os.path.join(self.folder, str(lec))
            if not fio.IsExistPath(lec_folder): continue
            
            for prompt in ['q1', 'q2', 'q3', 'POI', 'MP', 'LP']:
                prefix = os.path.join(lec_folder, prompt + '.sentence')
                if not fio.IsExist(prefix + phraseext): continue
                
                self.task_prefix.append(prefix)
                
                summaryfile = []
                for posfix in ['.ref.summary', '.ref.0', '.ref.1', '.ref.2', '.ref.3', '.ref.4', '.ref.5', '.ref.6', '.ref.7']:
                    sum_file = os.path.join(lec_folder, prompt + posfix)
                    if not fio.IsExist(sum_file): continue
                    summaryfile.append(sum_file)
                
                self.summary_files.append(summaryfile)
    
    def load_sumfiles(self):
        for sum_files in self.summary_files:
            sum_file_lines = []
            for sum_file in sum_files:
                doc = open(sum_file).readlines()
                sum_file_lines.append(doc)
            self.sumfiles.append(sum_file_lines)
    
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
            bigram_count = collections.defaultdict(int)
            for line in doc:
                bigram_tokens = ILP_getMC.ProcessLine(line, self.ngrams)
                
                for bi in bigram_tokens.split():
                    bigram_count[bi] += 1
                    
                bigrams.append(len(bigram_tokens.split()))
            self.bigrams.append(bigrams)
            self.bigram_counts.append(bigram_count)
    
    def extract_sum_bigram(self):
        for docs in self.sumfiles:
            sum_bigram = []
            for doc in docs:
                bigram_count = collections.defaultdict(int)
                for line in doc:
                    bigram_tokens = ILP_getMC.ProcessLine(line, self.ngrams)
                    
                    for bi in bigram_tokens.split():
                        bigram_count[bi] += 1

                sum_bigram.append(bigram_count)
            self.sum_bigram_counts.append(sum_bigram)
    
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
    
    def get_human_summary_bigram_percentage(self, N=0):
        if not self.sumfiles:
            self.load_sumfiles()
        
        if not self.sum_bigram_counts:
            self.extract_sum_bigram()
        
        total = 0.
        count_N = 0
        for bigram_counts, sum_bigram_counts in zip(self.bigram_counts, self.sum_bigram_counts):
            
            for bigram_lists in sum_bigram_counts:
                total += len(bigram_lists)
                for bigram in bigram_lists:
                    #check its count in keyfile
                    count = bigram_counts[bigram] if bigram in bigram_counts else 0
                    if count == N:
                        count_N += 1
                        #print bigram
                   
        return count_N/total
    
    def get_response_in_human_bigram_percentage(self, N=0):
        if not self.sumfiles:
            self.load_sumfiles()
        
        if not self.sum_bigram_counts:
            self.extract_sum_bigram()
        
        total = 0.
        count_N = 0
        for bigram_counts, sum_bigram_counts in zip(self.bigram_counts, self.sum_bigram_counts):
            
            for bigram, count in bigram_counts.items():
                if count != N: continue
                total += 1
                
                print bigram
                
                found = False
                for bigram_lists in sum_bigram_counts:
                    if bigram in bigram_lists:
                        found = True
                        break
                if found:
                    count_N += 1
                    
        return count_N/total

    def get_response_in_human_bigram_percentage_bigger(self, N=0):
        if not self.sumfiles:
            self.load_sumfiles()
        
        if not self.sum_bigram_counts:
            self.extract_sum_bigram()
        
        total = 0.
        count_N = 0
        for bigram_counts, sum_bigram_counts in zip(self.bigram_counts, self.sum_bigram_counts):
            
            total += len(bigram_counts)
            
            for bigram, count in bigram_counts.items():
                if count < N: continue
                for bigram_lists in sum_bigram_counts:
                    if bigram in bigram_lists:
                        count_N += 1
                   
        return count_N/total
    
    def get_human_summary_bigram_percentage_bigger(self, N=0):
        if not self.sumfiles:
            self.load_sumfiles()
        
        if not self.sum_bigram_counts:
            self.extract_sum_bigram()
        
        total = 0.
        count_N = 0
        for bigram_counts, sum_bigram_counts in zip(self.bigram_counts, self.sum_bigram_counts):
            
            for bigram_lists in sum_bigram_counts:
                total += len(bigram_lists)
                for bigram in bigram_lists:
                    #check its count in keyfile
                    count = bigram_counts[bigram] if bigram in bigram_counts else 0
                    if count >= N:
                        count_N += 1
                   
        return count_N/total
    
def get_summary(datadir, corpus, output):
    
    head = ['name', 'domain', 'task #', 'sen #', 'sen #/task', 'bigram #', 'bigram #/task', 'bigram #/sentence', 'size of A', 
            'sparsity of A', 'ratio of bigram >=2', 'Shannon index',
            'human in response N=0', 'human in response N=1', 'human in response N>1',
            'response in human N=1', 'response in human N=2', 'response in human N=3', 'response in human N=4', 'response in human N>=2',
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
        
        #output
        row.append(mc_corpus.get_human_summary_bigram_percentage(N=0))
        row.append(mc_corpus.get_human_summary_bigram_percentage(N=1))
        row.append(mc_corpus.get_human_summary_bigram_percentage_bigger(2))
        
        row.append(mc_corpus.get_response_in_human_bigram_percentage(N=1))
        row.append(mc_corpus.get_response_in_human_bigram_percentage(N=2))
        row.append(mc_corpus.get_response_in_human_bigram_percentage(N=3))
        row.append(mc_corpus.get_response_in_human_bigram_percentage(N=4))
        row.append(mc_corpus.get_response_in_human_bigram_percentage_bigger(N=2))
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