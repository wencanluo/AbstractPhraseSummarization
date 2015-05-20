import fio
import json
import sys
import porter
import NLTKWrapper
import os
import numpy
import NumpyWrapper

import ILP_baseline as ILP
import ILP_SVD

from feat_vec import FeatureVector

maxIter = 100

#Stemming
phraseext = ".key" #a list
studentext = ".keys.source" #json
countext = ".dict"  #a dictionary
lpext = ".lp"
lpsolext = ".sol"
sumexe = ".ref.summary"
featureext = ".f"

ngramTag = "___"

def LeaveOneLectureOutPermutation():
    sheets = range(0,12)
    N = len(sheets)
    for i in range(N):
        train = [str(k) for k in range(N) if k != i]
        #train = [str(i)]
        test = [str(i)]
        yield train, test

def generate_randomsummary(prefix, L, sumfile):
    print "no summary is found, generating random ones"
    lines = fio.ReadFile(prefix + phraseext)
    lines = [line.strip() for line in lines]
    
    index = numpy.random.permutation(len(lines))
    
    summaries = []
    
    length = 0
    for i in index:
        line = lines[i]
        length += len(line.split())
        
        if length <= L:
            summaries.append(line)
        else:
            length -= len(line.split())
    
    fio.SaveList(summaries, sumfile)
            
class ConceptWeightILP:
    def __init__(self, ilpdir, np, L, ngram, MalformedFlilter, featuredir, student_coverage, student_lambda, minthreshold, weight_normalization, no_training):
        self.ilpdir = ilpdir
        self.np = np
        self.L = L
        self.ngram = ngram
        self.MalformedFlilter = MalformedFlilter
        self.featuredir = featuredir
        self.student_coverage = student_coverage
        self.student_lambda = student_lambda
        self.minthreshold = minthreshold
        self.weight_normalization = weight_normalization
        self.no_training = no_training
        
        self.Weights = FeatureVector()
        self.WeightsNeg = FeatureVector()
        self.SumWeights = FeatureVector()
        self.SumWeightsNeg = FeatureVector()
        self.t = 0
        
    def run_crossvalidation(self):
        for train_lectures, test_lectures in LeaveOneLectureOutPermutation():
            if not self.no_training:
                self.train(train_lectures)
            self.test(train_lectures, test_lectures)
    
    def initialize_weight(self):
        self.Weights = FeatureVector()
        self.WeightsNeg = FeatureVector()
        self.SumWeights = FeatureVector()
        self.SumWeightsNeg = FeatureVector()
        self.t = 0

    def decode(self):
        prefix = self.prefix
        ngram = self.ngram
        MalformedFlilter = self.MalformedFlilter
        featurefile = self.featurefile
        student_coverage = self.student_coverage
        student_lambda = self.student_lambda
        minthreshold = self.minthreshold
        weight_normalization = self.weight_normalization
        
        # get each stemmed bigram, sequence the bigram and the phrase
        # bigrams: {index:bigram}, a dictionary of bigram index, X
        # phrases: {index:phrase}, is a dictionary of phrase index, Y
        #PhraseBigram: {phrase, [bigram]}
        self.IndexPhrase, self.IndexBigram, self.PhraseBigram = ILP.getPhraseBigram(prefix+phraseext, Ngram=ngram)
        fio.SaveDict(self.IndexPhrase, prefix + ".phrase_index.dict")
        fio.SaveDict(self.IndexBigram, prefix + ".bigram_index.dict")
        
        #get weight of bigrams {bigram:weigth}
        #BigramTheta = Weights #ILP.getBigramWeight_TF(PhraseBigram, phrases, prefix + countext) # return a dictionary
        
        #get word count of phrases
        self.PhraseBeta = ILP.getWordCounts(self.IndexPhrase)
        
        #get {bigram:[phrase]} dictionary
        self.BigramPhrase = ILP.getBigramPhrase(self.PhraseBigram)
    
        #get {student:phrase}
        #sequence students, students = {index:student}
        students, self.StudentPhrase = ILP.getStudentPhrase(self.IndexPhrase, prefix + studentext)
        fio.SaveDict(students, prefix + ".student_index.dict")
        
        #get {student:weight0}
        self.StudentGamma = ILP.getStudentWeight_One(self.StudentPhrase)
        
        self.FeatureVecU = LoadFeatureSet(featurefile)
        
        self.lpfile = prefix
        self.formulateProblem()
        
        m = ILP.SloveILP(self.lpfile)
        
        output = self.lpfile + '.L' + str(L) + ".summary"
        ILP.ExtractSummaryfromILP(self.lpfile, self.IndexPhrase, output)
    
    def get_round(self, train_lectures):
        round = 0
        for round in range(maxIter):
            weightfile = self.ilpdir + str(round) + '_' + '_'.join(train_lectures) + '_weight_' + "_" + '.json'
            if not fio.IsExist(weightfile):
                break
        
        if round != 0:
            nextround = round
            round = round -1
        else:
            nextround = 0
        
        return round, nextround
            
    def load_weight(self, train_lectures, round):
        if round == 0:
            self.initialize_weight()
        
        weightfile = self.ilpdir + str(round) + '_' + '_'.join(train_lectures) + '_weight_' + "_" + '.json'
        
        with open(weightfile, 'r') as fin:
            self.Weights = FeatureVector(json.load(fin, encoding="utf-8"))
    
    def train(self, train_lectures):
        self.stage = 'train'
        
        round, nextround = self.get_round(train_lectures)
        
        self.load_weight(train_lectures, round)
        
        for round in range(nextround, nextround+1):
            weightfile = self.ilpdir + str(round) + '_' + '_'.join(train_lectures) + '_weight_'  + "_" + '.json'
            bigramfile = ilpdir + str(round) + '_' + '_'.join(train_lectures) + '_bigram_'  + "_" + '.json'
        
            for sheet in train_lectures:
                week = int(sheet) + 1
                dir = self.ilpdir + str(week) + '/'
                
                for type in ['POI', 'MP', 'LP']:
                    self.prefix = dir + type + "." + np
                    self.featurefile = featuredir + str(week) + '/' + type + featureext
                    
                    print "update weight, round ", round
                    self.preceptron_update()
                    
            with open(weightfile, 'w') as fout:
                 json.dump(self.Weights, fout, encoding="utf-8",indent=2)
    
    def preceptron_update(self):
        #generate a system summary
        self.decode()
        
        #update Weights, and WeightsNeg
        #read the summary, update the weight 
        sumfile = self.prefix + '.L' + str(self.L) + '.summary'
    
        if len(fio.ReadFile(sumfile)) == 0:#no summary is generated, using a random baseline      
            generate_randomsummary(self.prefix, self.L, sumfile)
        
        _, System_IndexBigram, System_PhraseBigram = ILP.getPhraseBigram(self.prefix+phraseext, Ngram=self.ngram, MalformedFlilter=self.MalformedFlilter)
        
        #scan all the bigrams in the responses
        _, IndexBigram, Response_PhraseBigram = ILP.getPhraseBigram(self.prefix+phraseext, Ngram=self.ngram, MalformedFlilter=self.MalformedFlilter)
        
        reffile = ExtractRefSummaryPrefix(self.prefix) + '.ref.summary'
        _, Model_IndexBigram, Model_PhraseBigram = ILP.getPhraseBigram(reffile, Ngram=self.ngram, MalformedFlilter=self.MalformedFlilter)
        
        Model_BigramDict = getBigramDict(Model_IndexBigram, Model_PhraseBigram)
        
        #update the weights
        FeatureVecU = LoadFeatureSet(self.featurefile)
            
        pos = 0
        neg = 0
        correct_pos = 0
        correct_neg = 0
        
        for summary, bigrams in Response_PhraseBigram.items():
            for bigram in bigrams:
                bigramname = IndexBigram[bigram]
                if bigramname not in FeatureVecU: 
                    print bigramname
                    continue
                
                vec = FeatureVector(FeatureVecU[bigramname])
                y = 1.0 if bigramname in Model_BigramDict else -1.0
                
                if self.Weights.dot(vec)*y <= 0:
                    self.Weights += y*vec
                    
                    if y==1.0:
                        pos += 1
                    else:
                        neg += 1
                else:
                    if y==1.0:
                        correct_pos += 1
                    else:
                        correct_neg += 1
        
        print "pos:", pos
        print "neg:", neg
        print "correct_pos:", correct_pos
        print "correct_neg:", correct_neg
    
    def get_weight_product(self):
        BigramWeights = {}
        
        for bigram in self.BigramPhrase:
            bigramname = self.IndexBigram[bigram]
                    
            if bigramname in self.FeatureVecU:
                fvec = FeatureVector(self.FeatureVecU[bigramname])
                
                if self.stage=='train':
                    w = self.Weights.dot(fvec)
                else:#use averaged weight
                    w = self.Weights.dot(fvec)
                BigramWeights[bigram] = w
        
        median_w = numpy.median(BigramWeights.values())
        mean_w = numpy.mean(BigramWeights.values())
        std_w = numpy.std(BigramWeights.values())
        max_w = numpy.max(BigramWeights.values())
        min_w = numpy.min(BigramWeights.values())
                        
        if self.weight_normalization == 0:
            for bigram in BigramWeights:
                w = BigramWeights[bigram]
                BigramWeights[bigram] = w - self.minthreshold            
        elif self.weight_normalization == 1:#normalize to 0 ~ 1
            for bigram in BigramWeights:
                w = BigramWeights[bigram]
                if (max_w - min_w) != 0:
                    BigramWeights[bigram] = (w - min_w)/(max_w - min_w)
        elif self.weight_normalization == 2:#normalize to 0 ~ 1
            for bigram in BigramWeights:
                w = BigramWeights[bigram]
                if (max_w - mean_w - std_w) != 0:
                    BigramWeights[bigram] = (w - mean_w - std_w)/(max_w - mean_w - std_w)
        else:
            pass
                    
        return BigramWeights

    def formulateProblem(self):
        
        SavedStdOut = sys.stdout
        sys.stdout = open(self.lpfile + lpext, 'w')
    
        #write objective
        print "Maximize"
        objective = []
        
        BigramWeights = self.get_weight_product()
        
        if os.name == 'nt':
            import matplotlib.pyplot as plt
            plt.clf()
            plt.hist(BigramWeights.values(), bins=50)
            plt.savefig(self.lpfile + '.png')
            fio.SaveDict(BigramWeights, self.lpfile + '.bigram_weight.txt', SortbyValueflag=True)
        
        if self.student_coverage:
            for bigram in self.BigramPhrase:
                if bigram not in BigramWeights: 
                    print self.IndexBigram[bigram]
                    continue
                
                w = BigramWeights[bigram]
                
                if w <= 0: continue
                objective.append(" ".join([str(w*self.student_lambda), bigram]))
                         
            for student, grama in self.StudentGamma.items():
                if Lambda==1:continue
                 
                objective.append(" ".join([str(grama*(1-self.student_lambda)), student]))
        else:
            for bigram in self.BigramPhrase:
                if bigram not in BigramWeights: continue
                bigramname = self.IndexBigram[bigram]
                        
                w = BigramWeights[bigram]
                if w <= 0: continue
                objective.append(" ".join([str(w), bigram]))
        
        print "  ", " + ".join(objective)
        
        #write constraints
        print "Subject To"
        ILP.WriteConstraint1(self.PhraseBeta, L)
        
        ILP.WriteConstraint2(self.BigramPhrase)
        
        ILP.WriteConstraint3(self.PhraseBigram)
        
        if self.student_coverage:
            ILP.WriteConstraint4(self.StudentPhrase)
        
        indicators = []
        for bigram in self.BigramPhrase.keys():
            indicators.append(bigram)
        for phrase in self.PhraseBeta.keys():
            indicators.append(phrase)
            
        if self.student_coverage:
            for student in self.StudentGamma.keys():
                indicators.append(student)
            
        #write Bounds
        print "Bounds"
        for indicator in indicators:
            print "  ", indicator, "<=", 1
        
        #write Integers
        print "Integers"
        print "  ", " ".join(indicators)
        
        #write End
        print "End"
        sys.stdout = SavedStdOut
    
    def test(self, train_lectures, test_lectures):
        self.stage = 'test'
        
        round, nextround = self.get_round(train_lectures)
        self.load_weight(train_lectures, round)
            
        for sheet in test_lectures:
            week = int(sheet) + 1
            dir = ilpdir + str(week) + '/'
            
            for type in ['POI', 'MP', 'LP']:
                self.prefix = dir + type + "." + np
                print "Test: ", self.prefix
                
                self.featurefile = featuredir + str(week) + '/' + type + featureext
                self.decode()
                          

def LoadFeatureSet(featurename):
    with open(featurename, 'r') as fin:
        featureV = json.load(fin)
        
    return featureV
        
def getLastIndex(BigramIndex):
    maxI = 1
    for bigram in BigramIndex.values():
        if int(bigram[1:]) > maxI:
            maxI = int(bigram[1:])
    return maxI

def ExtractRefSummaryPrefix(prefix):
    key = prefix.rfind('.')
    if key==-1:
        return prefix
    return prefix[:key]

def getBigramDict(IndexBigram, PhraseBigram):
    dict = {}
    for phrase, bigrams in PhraseBigram.items():
        for bigram in bigrams:
            bigramname = IndexBigram[bigram]
            
            if bigramname not in dict:
                dict[bigramname] = 0
            dict[bigramname] = dict[bigramname] + 1
    return dict

def generate_randomsummary(prefix, L, sumfile):
    print "no summary is found, generating random ones"
    lines = fio.ReadFile(prefix + phraseext)
    lines = [line.strip() for line in lines]
    
    index = numpy.random.permutation(len(lines))
    
    summaries = []
    
    length = 0
    for i in index:
        line = lines[i]
        length += len(line.split())
        
        if length <= L:
            summaries.append(line)
        else:
            length -= len(line.split())
    
    fio.SaveList(summaries, sumfile)
    
if __name__ == '__main__':   
    
    from config import ConfigFile
    config = ConfigFile()
    
    ilpdir = "../../data/ILP_Sentence_Supervised_FeatureWeightingAveragePerceptron/"
    
    featuredir = ilpdir
    
    MalformedFlilter = False
    ngrams = config.get_ngrams()
    
    #for Lambda in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
    for Lambda in [config.get_student_lambda()]:
         #for L in [10, 15, 20, 25, 30, 35, 40, 45, 50]:
         for L in [config.get_length_limit()]:
             for np in ['sentence']: #'chunk
                 ilp = ConceptWeightILP(ilpdir, np, L, ngrams, MalformedFlilter, featuredir, 
                                         student_coverage = config.get_student_coverage(), 
                                         student_lambda = config.get_student_lambda(), 
                                         minthreshold=config.get_perceptron_threshold(), 
                                         weight_normalization=config.get_weight_normalization(), no_training=config.get_no_training())
                 for iter in range(config.get_perceptron_maxIter()):
                     ilp.run_crossvalidation()
    
    print "done"