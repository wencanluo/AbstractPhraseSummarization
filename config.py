import fio
import numpy

# dict_lambda = { 
#               1:
#                 {20: 4, 50: 4, 100:4, 200:2, 500:1.5, 1000:0.5, 2690:1
#                  },
#                2:
#                 {20: 3, 50: 4, 100:3, 200:2.5, 500:1.5, 1000:0.5, 2690:0
#                  },
#                3:
#                 {20: 4, 50: 4, 100:4, 200:2, 500:1.5, 1000:0.5, 2690:1
#                  },
#                4:
#                 {20: 3, 50: 4, 100:3, 200:2.5, 500:1.5, 1000:1, 2690:0
#                  },
#     }

dict_ngrams = {1:[1,2],
          2:[2],
          3:[1,2],
          4:[2],
          5:[2]
          }

class ConfigFile:
    def __init__(self, config_file_name = 'config.txt'):
        self.dict = fio.LoadDict(config_file_name, 'str')
    
    def get_response_split(self):
        v = self.dict['response_split']
        assert(v == '1' or v == '0')
        return v == '1'
    
    def set_response_split(self, v):
        self.dict['response_split'] = v
    
    def get_length_limit(self):
        L = int(self.dict['L'])
        assert(L >= 0 and L <= 100)
        return L
    
    def set_length_limit(self, L):
        self.dict['L'] = L
    
    def get_student_coverage(self):
        v = self.dict['student_coverage']
        assert(v == '1' or v == '0')
        return v == '1'
    
    def set_student_coverage(self, v):
        self.dict['student_coverage'] = v
        
    def get_student_lambda(self):
        if not self.get_student_coverage():
            return None
        
        v = float(self.dict['student_lambda'])
        assert(v >= 0 and v <= 1)
        return v
    
    def set_student_lambda(self, v):
        if not self.get_student_coverage():
            return
        self.dict['student_lambda'] = v
    
    def get_ngrams(self):
        v = self.dict['ngrams']
        assert(v == '1' or v == '2' or v == '3')
        
        if v == '1': #unigram
            return [1]
        if v == '2': #bigram
            return [2]
        if v == '3': #unigram + bigram
            return [1, 2]
        return [1, 2]
    
    def set_ngrams(self, v):
        self.dict['ngrams'] = v
    
    def get_features(self):
        v = self.dict['features']
        features = v.split(',')
        return [f.strip() for f in features]
    
    def set_features(self, v):
        self.dict['features'] = v
    
    def get_types(self):
        v = self.dict['types']
        types = v.split(',')
        return [f.strip() for f in types]
    
    def set_types(self, v):
        self.dict['types'] = v
    
    def get_position_bin(self):
        v = int(self.dict['position_bin'])
        assert(v >= 0 and v <= 10)
        return v
    
    def set_position_bin(self, v):
        self.dict['position_bin'] = v
        assert(v >= 0 and v <= 10)
        return v
    
    def get_perceptron_maxIter(self):
        v = int(self.dict['perceptron_maxIter'])
        return v
    
    def set_perceptron_maxIter(self, v):
        self.dict['perceptron_maxIter'] = v
        
    def get_perceptron_threshold(self):
        v = int(self.dict['perceptron_threshold'])
        return v
    
    def set_perceptron_threshold(self, v):
        self.dict['perceptron_threshold'] = v
        
    def get_weight_normalization(self):
        #0: no normalization, cutoff = minthreshold
        #1: normalization to {0, 1}
        #2: normalization to {0, 1} with a cutoff, (x - mean - std)/(max-mean-std)
        #else: raw weight
        v = int(self.dict['weight_normalization'])
        return v
    
    def set_weight_normalization(self, v):
        self.dict['weight_normalization'] = v
        
    def get_rank_max(self):
        softimpute_lambda = self.get_softImpute_lambda()
        
        assert(softimpute_lambda in numpy.arange(0.1, 4.1, 0.1))
        
        if softimpute_lambda >= 1.5:
            return 500
        else:
            return 2000
    
#     def set_rank_max(self, v):
#         self.dict['rank_max'] = v
            
    def get_binary_matrix(self):
        v = self.dict['binary_matrix']
        assert(v == '1' or v == '0')
        return v == '1'
    
    def set_binary_matrix(self, v):
        self.dict['binary_matrix'] = v
        
    def get_exp_id(self):
        v = int(self.dict['exp_id'])
        
        if v==1:
            assert(self.get_ngrams() == [1,2])
            #assert(self.get_binary_matrix() == False)
        elif v==2:
            assert(self.get_ngrams() == [2])
            #assert(self.get_binary_matrix() == False)
        elif v==3:
            assert(self.get_ngrams() == [1,2])
            #assert(self.get_binary_matrix() == True)
        elif v==4 or v==5:
            assert(self.get_ngrams() == [2])
            #assert(self.get_binary_matrix() == True)
            
        return v
    
    def set_exp_id(self, v):
        self.dict['exp_id'] = v
        
    def get_matrix_dir(self):
        v = self.dict['matrix_dir']
        return v + 'exp' + str(self.get_exp_id()) + '/'
    
    def set_matrix_dir(self, v):
        self.dict['matrix_dir'] = v
        
    def get_softImpute_lambda(self):
        v = float(self.dict['softInpute_lambda'])
        return v
    
    def set_softImpute_lambda(self, v):
        self.dict['softInpute_lambda'] = v
        
    def get_sparse_threshold(self):
        v = float(self.dict['sparse_threshold'])
        return v
    
    def set_sparse_threshold(self, v):
        self.dict['sparse_threshold'] = v
    
    def get_no_training(self):
        v = self.dict['no_training']
        assert(v == '1' or v == '0')
        return v == '1'
    
    def get_prefixA(self):
        rank = self.get_rank_max()
        Lambda = self.get_softImpute_lambda()
        if rank == 0:
            prefixA = '.org.softA'
        else:
            prefixA = '.' + str(rank) + '_' + str(Lambda) + '.softA'
        return prefixA
    
    def get_lcs_ratio(self):
        v = float(self.dict['lcs_ratio'])
        return v
                                    
    def toString(self):
        s= ""
        for k, v in self.dict.items():
            s += str(k) + '_' + str(v) + '@'
        return s
    
    def save(self):
        import sys
        SavedStdOut = sys.stdout
        sys.stdout = open('config.txt', 'w')
        
        keys = [
            'L', 
            'student_coverage', 
            'student_lambda', 
            'ngrams', 
            'features', 
            'position_bin', 
            'perceptron_threshold', 
            'weight_normalization', 
            'perceptron_maxIter', 
            'matrix_dir', 
            'types',
            'sparse_threshold', 
            'softInpute_lambda', 
            'exp_id',
            'no_training',
            'lcs_ratio',
            ]
        
        for key in keys:
            print str(key) + "\t" + str(self.dict[key])
        
        sys.stdout = SavedStdOut
        
if __name__ == '__main__':
    config = ConfigFile()
    print config.get_response_split()
    print config.toString()