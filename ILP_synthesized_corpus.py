import os, fio
import ILP_getMatrixCompletion as ILP_getMC
import ILP_MC
import numpy as np
import json
from scipy.stats import entropy
import collections
import global_params
import random

phraseext = ".key" #a list
studentext = ".keys.source" #json
countext = ".dict"  #a dictionary
orgA_ext = '.org.softA'

class MCCorpus:
    def __init__(self, folder, cid):
        self.folder= os.path.join(folder, 'ILP_MC')
        self.mc_folder = os.path.join(folder, 'MC')
        self.cid = cid
        
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
        if not self.keyfiles:
            self.load_keyfiles()
            
        for doc in self.keyfiles:
            bigrams = []
            bigram_count = {}
            for i, line in enumerate(doc):
                bigram_tokens = ILP_getMC.ProcessLine(line, self.ngrams)
                
                for bi in bigram_tokens.split():
                    if bi not in bigram_count:
                        bigram_count[bi] = {}
                        bigram_count[bi]['count'] = 0
                        bigram_count[bi]['s'] = set()
                        
                    bigram_count[bi]['count'] += 1
                    bigram_count[bi]['s'].add(i)
                    
                bigrams.append(len(bigram_tokens.split()))
            self.bigrams.append(bigrams)
            self.bigram_counts.append(bigram_count)
    
    def extract_bigrams_filter(self):
        '''
        extract the bigram counts in student responses
        '''
        if not self.keyfiles:
            self.load_keyfiles()
        
        self.bigram_counts_filter = []
        for doc, filters in zip(self.keyfiles, self.filters):
            bigrams = []
            bigram_count = collections.defaultdict(int)
            for i, line in enumerate(doc):
                if i in filters: continue
                
                bigram_tokens = ILP_getMC.ProcessLine(line, self.ngrams)
                
                for bi in bigram_tokens.split():
                    bigram_count[bi] += 1
                
            self.bigram_counts_filter.append(bigram_count)
    
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
    
    def get_bigram_num(self):
        if not self.bigrams:
            self.extract_bigrams()
        self.bigram_num = np.sum(np.sum(self.bigrams)) 
        return self.bigram_num
    
    def get_bigram_num_per_task(self):
        return '%.1f'% (self.bigram_num/float(self.task_num))
    
    def get_bigram_num_per_sentence(self):
        return '%.1f'%(self.bigram_num/float(self.total_number_of_sentences))
    
    def get_human_summary_bigram_percentage_increase(self, N=0, K=1, outputdir=None):
        '''
        increase the ratio by deleting responses with N>1
        '''
        
        if not self.bigrams:
            self.extract_bigrams()
        
        if not self.sumfiles:
            self.load_sumfiles()
        
        if not self.sum_bigram_counts:
            self.extract_sum_bigram()
        
        #IE256, K=1->21.0, K=10->31.2
        #IE256_2016, K=1->23.7, K=5->29.9, K=10->31.2
        
        #review_camera, K=1->84.9, K=5->85.8  K=10->86.2
        #DUC04, K=1->16.0, K=5->16.5, K=10->17.2, K=20->18.4, K=50->21.2, K=100->23.4
        #CS0445, K=1->28.0, K=5->32.7, K=10->34.2
        #Engineer, K=1->36.0, K=5->38.6, K=10->41.4
        #review_prHistory, K=1->77.4, K=5->78.7, K=10->80.4
        #review_IMDB, K=1->76.5, K=10->76.8
        #K = 10 #remove one sentence
        
        self.filters = []
        
        total = 0.
        count_N = 0
        doc_id = 0 #document id
        for bigram_counts, sum_bigram_counts in zip(self.bigram_counts, self.sum_bigram_counts):
            once = set() #responses that contain the bigram only once
            more = set() 
            for bigram_lists in sum_bigram_counts:
                total += len(bigram_lists)
                for bigram in bigram_lists:
                    #check its count in keyfile
                    if bigram not in bigram_counts: continue
                    
                    count = bigram_counts[bigram]['count'] if bigram in bigram_counts else 0
                    if count == N:
                        count_N += 1
                        #print bigram
                        
                        for ss in bigram_counts[bigram]['s']:
                            once.add(ss) 
                    else:
                        for ss in bigram_counts[bigram]['s']:
                            more.add(ss) 
                        
            for bi in once:
                more.discard(bi)
             
            #randomly remove sentences
            more = list(more)
            random.shuffle(more)
            
            self.filters.append(more[:K])
            
            doc_id += 1
        
        #recount the value
        self.extract_bigrams_filter()
        
        total = 0.
        count_N = 0
        for bigram_counts, sum_bigram_counts in zip(self.bigram_counts_filter, self.sum_bigram_counts):
            for bigram_lists in sum_bigram_counts:
                total += len(bigram_lists)
                for bigram in bigram_lists:
                    #check its count in keyfile
                    count = bigram_counts[bigram] if bigram in bigram_counts else 0
                    if count == N:
                        count_N += 1
                        #print bigram
                   
        ratio = '%2.1f'%(count_N/total*100)
        
#         outdirs = [
#                self.folder.replace(self.cid, '%s_%s'%(self.cid, ratio)),
#                self.mc_folder.replace(self.cid, '%s_%s'%(self.cid, ratio)),
#                ]
#         
#         for outdir in outdirs:

        outpath = self.folder.replace(self.cid, '%s_%s'%(self.cid, ratio))
        print outpath
        
        fio.NewPath(outpath)
        
        #write the new text file
        for prefix, doc, filters in zip(self.task_prefix, self.keyfiles, self.filters):
            
            student_file = prefix + studentext
            source = fio.LoadDictJson(student_file)
            
            count_file = prefix + countext
            CountDict = fio.LoadDict(count_file, 'float')
            
            prefix_file = prefix.replace('/', '\\').split('\\')
            
            prefix = '\\'.join(prefix_file[:-1]).replace(self.cid, '%s_%s'%(self.cid, ratio))
            suffix = prefix_file[-1]
            
            suffix = suffix.replace('POI', 'q1')
            suffix = suffix.replace('MP', 'q2')
            suffix = suffix.replace('LP', 'q3')
            
            fio.NewPath(prefix)
            
            keysentences = []
            for i, line in enumerate(doc):
                if i in filters: continue
                keysentences.append(line.rstrip('\r\n'))
            
            outout = os.path.join(prefix, suffix+phraseext)
            fio.SaveList(keysentences, outout)
             
            print outout
             
            output = os.path.join(prefix, suffix + studentext)
            fio.SaveDict2Json(source, output)
                 
            output = os.path.join(prefix, suffix + countext)
            fio.SaveDict(CountDict, output)
             
            #MC
            prefix = prefix.replace('ILP_MC', 'MC')
            fio.NewPath(prefix)
             
            outout = os.path.join(prefix, suffix+phraseext)
            fio.SaveList(keysentences, outout)
             
            output = os.path.join(prefix, suffix + studentext)
            fio.SaveDict2Json(source, output)
                 
            output = os.path.join(prefix, suffix + countext)
            fio.SaveDict(CountDict, output)
        
        return count_N/total
    
    def get_human_summary_bigram_percentage_decrease(self, N=0, K=1, outputdir=None):
        '''
        decrease the ratio by deleting responses with N=1
        '''
        
        if not self.bigrams:
            self.extract_bigrams()
        
        if not self.sumfiles:
            self.load_sumfiles()
        
        if not self.sum_bigram_counts:
            self.extract_sum_bigram()
        
        #IE256, K=1->11.9, K5=5.6, K=10->5.4
        #IE256_2016, K=1->13.2, K=5->5.4
        
        #DUC04, K=1->15.5, K=5->13.9, K=10->12.6
        #CS0445, K=1->19.3, K=5->11.0
        #Engineer, K=1->26.5, K=5->16.0, K=10->15.2
        
        #review_camera, K=1->83.2, K=5->78.7  K=10->74.5, K=20->74.3
        #review_prHistory, K=1->75.6, K=5->71.3
        #review_IMDB, K=1->74.8, K=10->70.8
        #K = 5 #remove one sentence
        
        self.filters = []
        
        total = 0.
        count_N = 0
        doc_id = 0 #document id
        for bigram_counts, sum_bigram_counts in zip(self.bigram_counts, self.sum_bigram_counts):
            once = set() #responses that contain the bigram only once
            more = set()
            for bigram_lists in sum_bigram_counts:
                total += len(bigram_lists)
                for bigram in bigram_lists:
                    #check its count in keyfile
                    if bigram not in bigram_counts: continue
                    
                    count = bigram_counts[bigram]['count'] if bigram in bigram_counts else 0
                    if count == N:
                        count_N += 1
                        #print bigram
                        
                        for ss in bigram_counts[bigram]['s']:
                            once.add(ss) 
                    else:
                        for ss in bigram_counts[bigram]['s']:
                            more.add(ss) 
                        
            for bi in more:
                once.discard(bi)
             
            #randomly remove sentences
            once = list(once)
            random.shuffle(once)
            
            self.filters.append(once[:K])
            
            doc_id += 1
        
        #recount the value
        self.extract_bigrams_filter()
        
        total = 0.
        count_N = 0
        for bigram_counts, sum_bigram_counts in zip(self.bigram_counts_filter, self.sum_bigram_counts):
            for bigram_lists in sum_bigram_counts:
                total += len(bigram_lists)
                for bigram in bigram_lists:
                    #check its count in keyfile
                    count = bigram_counts[bigram] if bigram in bigram_counts else 0
                    if count == N:
                        count_N += 1
                        #print bigram
                   
        ratio = '%2.1f'%(count_N/total*100)
        
#         outdirs = [
#                self.folder.replace(self.cid, '%s_%s'%(self.cid, ratio)),
#                self.mc_folder.replace(self.cid, '%s_%s'%(self.cid, ratio)),
#                ]
#         
#         for outdir in outdirs:

        outpath = self.folder.replace(self.cid, '%s_%s'%(self.cid, ratio))
        print outpath
        
        fio.NewPath(outpath)
        
        #write the new text file
        for prefix, doc, filters in zip(self.task_prefix, self.keyfiles, self.filters):
            
            student_file = prefix + studentext
            source = fio.LoadDictJson(student_file)
            
            count_file = prefix + countext
            CountDict = fio.LoadDict(count_file, 'float')
            
            prefix_file = prefix.replace('/', '\\').split('\\')
            
            prefix = '\\'.join(prefix_file[:-1]).replace(self.cid, '%s_%s'%(self.cid, ratio))
            suffix = prefix_file[-1]
            
            suffix = suffix.replace('POI', 'q1')
            suffix = suffix.replace('MP', 'q2')
            suffix = suffix.replace('LP', 'q3')
            
            fio.NewPath(prefix)
            
            keysentences = []
            for i, line in enumerate(doc):
                if i in filters: continue
                keysentences.append(line.rstrip('\r\n'))
            
            outout = os.path.join(prefix, suffix+phraseext)
            fio.SaveList(keysentences, outout)
             
            print outout
             
            output = os.path.join(prefix, suffix + studentext)
            fio.SaveDict2Json(source, output)
                 
            output = os.path.join(prefix, suffix + countext)
            fio.SaveDict(CountDict, output)
             
            #MC
            prefix = prefix.replace('ILP_MC', 'MC')
            fio.NewPath(prefix)
             
            outout = os.path.join(prefix, suffix+phraseext)
            fio.SaveList(keysentences, outout)
             
            output = os.path.join(prefix, suffix + studentext)
            fio.SaveDict2Json(source, output)
                 
            output = os.path.join(prefix, suffix + countext)
            fio.SaveDict(CountDict, output)
        
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
                
                #print bigram
                
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
    
    for name, domain, K1, K2 in corpus:
        folder = os.path.join(datadir, name)
        print folder
        random.seed(1234567)
        
        mc_corpus1 = MCCorpus(folder, name)
        mc_corpus1.load_task()
        print mc_corpus1.get_human_summary_bigram_percentage_increase(N=1, K=K1)
    
        random.seed(1234567)
        mc_corpus2 = MCCorpus(folder, name)
        mc_corpus2.load_task()
        print mc_corpus2.get_human_summary_bigram_percentage_decrease(N=1, K=K2)
        
if __name__ == '__main__':
    
    #increase
    #IE256, K=1->21.0, K=10->31.2
    #IE256_2016, K=1->23.7, K=5->29.9, K=10->31.2
    
    #review_camera, K=1->84.9, K=5->85.8  K=10->86.2
    #DUC04, K=1->16.0, K=5->16.5, K=10->17.2, K=20->18.4, K=50->21.2, K=100->23.4
    #CS0445, K=1->28.0, K=5->32.7, K=10->34.2
    #Engineer, K=1->36.0, K=5->38.6, K=10->41.4
    #review_prHistory, K=1->77.4, K=5->78.7, K=10->80.4
    #review_IMDB, K=1->76.5, K=10->76.8
    
    #decrease
    #IE256, K=1->11.9, K5=5.6, K=10->5.4
    #IE256_2016, K=1->13.2, K=5->5.4
    
    #DUC04, K=1->15.5, K=5->13.9, K=10->12.6
    #CS0445, K=1->19.3, K=5->11.0
    #Engineer, K=1->26.5, K=5->16.0, K=10->15.2
    
    #review_camera, K=1->83.2, K=5->78.7  K=10->74.5, K=20->74.3
    #review_prHistory, K=1->75.6, K=5->71.3
    #review_IMDB, K=1->74.8, K=10->70.8
    
    datadir = '../../data/'
    corpus = [
#             ('Engineer','response'),
#             ('IE256','response', 1, 1),
#             ('IE256_2016','response', 1, 1),
#             ('CS0445','response', 10, 5),
            ('review_camera','review', 5, 5),
            ('review_prHistory','review', 5,5),
            ('review_IMDB','review', 10, 5),
#             ('DUC04', 'news'),
#             ('TAC_s08', 'news'),
#             ('TAC_s09', 'news'),
#             ('TAC_s10', 'news'),
#             ('TAC_s11', 'news'),
              ]
    
    output = '../../data/statistcis_all_news.txt'
    get_summary(datadir, corpus, output)