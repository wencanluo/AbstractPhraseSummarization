import os, fio
import ILP_getMatrixCompletion as ILP_getMC
import ILP_MC
import numpy as np
import json
from scipy.stats import entropy
import collections
import global_params

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
        '''
        read the task prefix and human summary files
        self.summary_files: [[summary filename]]
        '''
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
        '''
        load the human summary file into self.sumfiles [[[lines]]]
        '''
        for sum_files in self.summary_files:
            sum_file_lines = []
            for sum_file in sum_files:
                doc = open(sum_file).readlines()
                sum_file_lines.append(doc)
            self.sumfiles.append(sum_file_lines)
    
    def load_keyfiles(self):
        '''
        load the sentence files
        '''
        for prefix in self.task_prefix:
            key_file = prefix + phraseext
            
            doc = open(key_file).readlines()
            self.keyfiles.append(doc)
    
    def get_number_of_ave_authors(self):
        '''
        get the number of averaged authors for tasks
        '''
        
        N = []
        for prefix in self.task_prefix:
            source_file = prefix + studentext
            sources = fio.LoadDictJson(source_file)
            
            students = set()
            for key in sources:
                if isinstance(sources[key], list):
                    for stu in sources[key]:
                        students.add(stu)
                else:
                    students.add(sources[key])
            N.append(len(students))
            
        return np.mean(N)
        
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
    
    def get_num_words_per_task(self):
        self.wc = []
        for doc in self.keyfiles:
            count = 0
            for line in doc:
                count += len(line.split())
            self.wc.append(count)
        return '%.1f'%np.mean(self.wc)
	
    def get_num_words_per_sentence(self):
        self.wc_sen = []
        for doc in self.keyfiles:
            for line in doc:
                self.wc_sen.append(len(line.split()))
        return '%.1f'%np.mean(self.wc_sen)
    
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
        self.get_number_of_sentence()
        print self.total_number_of_sentences
        print self.bigram_num
        return self.total_number_of_sentences * long(self.bigram_num)
    
    def load_A(self):
        self.As = []
        
        for doc in self.keyfiles:
            A = collections.defaultdict(list)
            
            bigrams = []
            bigram_set = set()
            for line in doc:
                bigram_tokens = ILP_getMC.ProcessLine(line, self.ngrams)
                
                B = bigram_tokens.split()
                B = set(B)
                
                bigrams.append(B)
                
                for b in B:
                    bigram_set.add(b)
            
            for B in bigrams:
                for b in bigram_set:
                    if b in B:
                        A[b].append(1.0)
                    else:
                        A[b].append(0.0)
            self.As.append(A)
        
    def load_A_MC(self):
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
    
    def get_bigram_ratio_eqK_A(self, A, K=1):
        total = float(len(A))
        count = 0
        for bigram in A:
            if sum(A[bigram]) == K:
                count += 1
        return count/total
    
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
        
        return '%.1f%%'%(np.mean(self.sparsities)*100)
    
    def get_bigram_ratio_more_than_eqK(self, K):
        self.bigram_ratio_eqNs = []
        
        for A in self.As:
            self.bigram_ratio_eqNs.append(self.get_bigram_ratio_eqK_A(A, K))
        
        return '%.1f%%'%(np.mean(self.bigram_ratio_eqNs)*100)
    
    def get_bigram_ratio_more_than_N(self):
        self.bigram_ratio_Ns = []
        
        for A in self.As:
            self.bigram_ratio_Ns.append(self.get_bigram_ratio_N_A(A))
        
        return '%.1f%%'%(np.mean(self.bigram_ratio_Ns)*100)
    
    
    def get_bigram_entropy(self):
        self.bigram_entropies = []
        
        for A in self.As:
            self.bigram_entropies.append(self.get_bigram_entropy_A(A))
        
        return '%.3f'%(np.mean(self.bigram_entropies))
    
    def get_human_summary_length(self):
        '''
        Extract the average human summary length in number of words
        '''
        if not self.sumfiles:
            self.load_sumfiles()
        
        self.sum_lengthes = []
        for sun_file_lines in self.sumfiles:
            ave = []
            for lines in sun_file_lines:
                length = 0
                for line in lines:
                    tokens = line.lower().split()
                    length += len(tokens)
                ave.append(length)
            self.sum_lengthes.append(np.mean(ave))
        return '%.0f'%(np.mean(self.sum_lengthes))
    
    def get_compression_ratio(self):
        self.compression_ratio = []
        for length, wc in zip(self.sum_lengthes, self.wc):
            self.compression_ratio.append(float(length)/wc)
        return '%.3f'%(np.mean(self.compression_ratio))
    
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
                   
        return '%.1f%%'%(count_N/total*100)
    
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
                
                #print bigram
                
                found = False
                for bigram_lists in sum_bigram_counts:
                    if bigram in bigram_lists:
                        found = True
                        break
                if found:
                    count_N += 1
                    
        return '%.1f%%'%(count_N/total*100)

    def get_response_in_human_bigram_percentage_bigger(self, N=0):
        if not self.sumfiles:
            self.load_sumfiles()
        
        if not self.sum_bigram_counts:
            self.extract_sum_bigram()
        
        total = 0.
        count_N = 0
        for bigram_counts, sum_bigram_counts in zip(self.bigram_counts, self.sum_bigram_counts):
            for bigram, count in bigram_counts.items():
                if count < N: continue
                
                total += 1
                found = False
                for bigram_lists in sum_bigram_counts:
                    if bigram in bigram_lists:
                        found = True
                        break
                if found:
                    count_N += 1
                   
        return '%.1f%%'%(count_N/total*100)
    
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
                   
        return '%.1f%%'%(count_N/total*100)
    
def get_summary(datadir, corpus, output):
    
    head = ['name', 'genre', 'T', 'au', 
            'M*N', 'M', 'N', 'M/T', 'N/T', 'N/M', 'W/T', 'W/M',
            's', '$b=1$', '$b>1$', 'H',
            'L', 'hs', 'r', '$\\alpha_{b>0}$', '$\\alpha_{b=0}$', '$\\alpha_{b=1}$', '$\\alpha_{b>1}$',
            '$\\beta_{b=1}$', '$\\beta_{b=2}$', '$\\beta_{b=3}$', '$\\beta_{b=4}$', '$\\beta_{b>1}$',
            ]
    
    ids = ['id'] + range(1, len(head))
    print ids
    
    body = [head]
    
    for name, domain in corpus:
        print name
        
        row = [global_params.mapcid(name), domain]  #1
        folder = os.path.join(datadir, name)
        
        mc_corpus = MCCorpus(folder)
        mc_corpus.load_task()

        row.append(mc_corpus.get_task_num()) #2
        row.append('%.1f'%mc_corpus.get_number_of_ave_authors()) #3
        
        M = mc_corpus.get_number_of_sentence()
        N = mc_corpus.get_bigram_num()
        row.append(mc_corpus.get_size()) #4
        row.append(M) #5
        row.append(N) #6
        
        row.append(mc_corpus.get_number_of_sentence_per_task()) #7
        
        row.append(mc_corpus.get_bigram_num_per_task()) #8
        row.append(mc_corpus.get_bigram_num_per_sentence()) #9
        row.append(mc_corpus.get_num_words_per_task()) #10
        row.append(mc_corpus.get_num_words_per_sentence()) #10.5
        
          
        row.append(mc_corpus.get_sparsity_ratio()) #11
        row.append(mc_corpus.get_bigram_ratio_more_than_eqK(1)) #12
        
        row.append(mc_corpus.get_bigram_ratio_more_than_N()) #13
        row.append(mc_corpus.get_bigram_entropy()) #14

        #output
        row.append(mc_corpus.get_human_summary_length()) #15        
        row.append(global_params.no_human[name]) #16
        row.append(mc_corpus.get_compression_ratio()) #17
        row.append(mc_corpus.get_human_summary_bigram_percentage_bigger(1)) #18
        row.append(mc_corpus.get_human_summary_bigram_percentage(N=0)) #19
        row.append(mc_corpus.get_human_summary_bigram_percentage(N=1)) #20
#         row.append(mc_corpus.get_human_summary_bigram_percentage(N=2))
#         row.append(mc_corpus.get_human_summary_bigram_percentage(N=3))
#         row.append(mc_corpus.get_human_summary_bigram_percentage(N=4))
#         row.append(mc_corpus.get_human_summary_bigram_percentage(N=5))
        row.append(mc_corpus.get_human_summary_bigram_percentage_bigger(2)) #21
          
        row.append(mc_corpus.get_response_in_human_bigram_percentage(N=1)) #22
        row.append(mc_corpus.get_response_in_human_bigram_percentage(N=2)) #23
        row.append(mc_corpus.get_response_in_human_bigram_percentage(N=3))#24
        row.append(mc_corpus.get_response_in_human_bigram_percentage(N=4)) #25
        row.append(mc_corpus.get_response_in_human_bigram_percentage_bigger(N=2)) #26
        body.append(row)
    
    fio.Write2LatexTrans(output, body, ids)
        
if __name__ == '__main__':
    
    datadir = '../../data/'
    corpus = [
            ('DUC04', 'news'),
            ('review_camera','review'),
            ('review_IMDB','review'),
            ('review_prHistory','review'),
            ('Engineer','response'),
            ('IE256','response'),
            ('IE256_2016','response'),
            ('CS0445','response')
#             ('TAC_s08', 'news'),
#             ('TAC_s09', 'news'),
#             ('TAC_s10', 'news'),
#             ('TAC_s11', 'news'),
              ]
    
    output = '../../data/statistcis_all_news.txt'
    get_summary(datadir, corpus, output)