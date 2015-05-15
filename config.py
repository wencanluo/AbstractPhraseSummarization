import fio

class ConfigFile:
    def __init__(self, config_file_name = 'config.txt'):
        self.dict = fio.LoadDict(config_file_name, 'str')
    
    def get_response_split(self):
        v = self.dict['response_split']
        assert(v == '1' or v == '0')
        return v == '1'
    
    def get_length_limit(self):
        L = int(self.dict['L'])
        assert(L >= 0 and L <= 100)
        return L
    
    def get_student_coverage(self):
        v = self.dict['student_coverage']
        assert(v == '1' or v == '0')
        return v == '1'
    
    def get_student_lambda(self):
        if not self.get_student_coverage():
            return None
        
        v = float(self.dict['student_lambda'])
        assert(v >= 0 and v <= 1)
        return v
    
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
    
    def get_features(self):
        v = self.dict['features']
        features = v.split(',')
        return [f.strip() for f in features]
    
    def get_position_bin(self):
        v = int(self.dict['position_bin'])
        assert(v >= 0 and v <= 10)
        return v
    
    def get_perceptron_maxIter(self):
        v = int(self.dict['perceptron_maxIter'])
        return v
    
    def get_perceptron_threshold(self):
        v = int(self.dict['perceptron_threshold'])
        return v
    
    def get_weight_normalization(self):
        v = self.dict['weight_normalization']
        assert(v == '1' or v == '0')
        return v == '1'
            
    def toString(self):
        s= ""
        for k, v in self.dict.items():
            s += str(k) + '_' + str(v) + '@'
        return s

if __name__ == '__main__':
    config = ConfigFile()
    print config.get_response_split()
    print config.toString()