import fio
import json
import sys
import porter
import NLTKWrapper
import os
import numpy
import NumpyWrapper
import math

import scipy

import ILP_baseline as ILP

from feat_vec import FeatureVector

maxIter = 1000

#Stemming
phraseext = ".key" #a list
studentext = ".keys.source" #json
countext = ".dict"  #a dictionary
lpext = ".lp"
lpsolext = ".sol"
sumexe = ".ref.summary"
featureext = ".f"

ngramTag = "___"

stage_train = 0
stage_test =1

def LeaveOneLectureOutPermutation():
    sheets = range(0,12)
    #sheets = range(0,1)
    for i in sheets:
        train = [str(k) for k in range(len(sheets)) if k != i]
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
    def __init__(self, ilpdir, np, L, ngram, MalformedFlilter, featuredir, student_coverage, student_lambda, 
                 minthreshold, weight_normalization, no_training, types):
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
        self.types = types
        
        self.Weights = FeatureVector()
        self.WeightsNeg = FeatureVector()
        self.SumWeights = FeatureVector()
        self.SumWeightsNeg = FeatureVector()
        self.t = 0
    
    def gather_rouges(self):
        Header = ['method', 'iter'] + ['R1-R', 'R1-P', 'R1-F', 'R2-R', 'R2-P', 'R2-F', 'RSU4-R', 'RSU4-P', 'RSU4-F']*3
        body = []
        
        round, _ = self.get_round(self.train_lectures)
        
        for i in range(0, round+1):
            row = ['baseline+CW']
            row.append(i)
            
            rougename = self.ilpdir+'rouge.sentence.L' +str(config.get_length_limit())+ '.w' + str(self.weight_normalization) +'.r'+ str(i) + ".txt"
            
            scores = ILP.getRouges(rougename)
            
            row = row + scores
            body.append(row)
        
        newname = self.ilpdir+'rouge.sentence.L' +str(config.get_length_limit())+ '.w' + str(self.weight_normalization) + ".txt"
        fio.WriteMatrix(newname, body, Header)
        
    def run_crossvalidation(self):
        for train_lectures, test_lectures in LeaveOneLectureOutPermutation():
            if not self.no_training:
                self.train(train_lectures)
        
        for train_lectures, test_lectures in LeaveOneLectureOutPermutation():
            self.test(train_lectures, test_lectures)
        
        round, _ = self.get_round(self.train_lectures)
        
        rougename = self.ilpdir+'rouge.sentence.L' +str(config.get_length_limit())+ '.w' + str(self.weight_normalization) +'.r'+ str(round) + ".txt"
        os.system('python ILP_GetRouge.py '+self.ilpdir)
                
        rougefile = self.ilpdir + "rouge.sentence.L"+str(config.get_length_limit())+".txt"
        os.system('mv ' + rougefile + ' ' + rougename)
    
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
        self.formulate_problem()
        
        m = ILP.SloveILP(self.lpfile)
        
        output = self.lpfile + '.L' + str(L) + ".summary"
        
        fio.remove(output)
        
        ILP.ExtractSummaryfromILP(self.lpfile, self.IndexPhrase, output)
    
    def get_round(self, train_lectures):
        round = 0
        for round in range(maxIter):
            weightfile = self.ilpdir + str(round) + '_' + '_'.join(train_lectures) + '_weight' + "_" + '.json'
            if not fio.IsExist(weightfile):
                break
        
        if round != 0:
            nextround = round
            round = round -1
        else:
            nextround = 0
        
        return round, nextround
            
    def load_weight(self, train_lectures, round):
        weightfile = self.ilpdir + str(0) + '_' + '_'.join(train_lectures) + '_weight' + "_" + '.json'
        if not fio.IsExist(weightfile):
            self.initialize_weight()
            return
        
        weightfile = self.ilpdir + str(round) + '_' + '_'.join(train_lectures) + '_weight' + "_" + '.json'
        with open(weightfile, 'r') as fin:
            self.Weights = FeatureVector(json.load(fin, encoding="utf-8"))
            
        weightfile = self.ilpdir + str(round) + '_' + '_'.join(train_lectures) + '_weight_neg' + "_" + '.json'
        with open(weightfile, 'r') as fin:
            self.WeightsNeg = FeatureVector(json.load(fin, encoding="utf-8"))
            
        weightfile = self.ilpdir + str(round) + '_' + '_'.join(train_lectures) + '_weight_sum' + "_" + '.json'
        with open(weightfile, 'r') as fin:
            self.SumWeights = FeatureVector(json.load(fin, encoding="utf-8"))
            
        weightfile = self.ilpdir + str(round) + '_' + '_'.join(train_lectures) + '_weight_neg_sum' + "_" + '.json'
        with open(weightfile, 'r') as fin:
            self.SumWeightsNeg = FeatureVector(json.load(fin, encoding="utf-8"))
        
        tfile = self.ilpdir + str(round) + '_' + '_'.join(train_lectures) + '_t'+ '.json'
        with open(tfile, 'r') as fin:
            self.t = json.load(fin, encoding="utf-8")
    
    def save_weight(self, train_lectures, round):
        weightfile = self.ilpdir + str(round) + '_' + '_'.join(train_lectures) + '_weight'  + "_" + '.json'
        with open(weightfile, 'w') as fout:
            json.dump(self.Weights, fout, encoding="utf-8",indent=2)
        
        weightfile = self.ilpdir + str(round) + '_' + '_'.join(train_lectures) + '_weight_neg'  + "_" + '.json'
        with open(weightfile, 'w') as fout:
            json.dump(self.WeightsNeg, fout, encoding="utf-8",indent=2)
        
        weightfile = self.ilpdir + str(round) + '_' + '_'.join(train_lectures) + '_weight_sum'  + "_" + '.json'
        with open(weightfile, 'w') as fout:
            json.dump(self.SumWeights, fout, encoding="utf-8",indent=2)
        
        weightfile = self.ilpdir + str(round) + '_' + '_'.join(train_lectures) + '_weight_neg_sum'  + "_" + '.json'
        with open(weightfile, 'w') as fout:
            json.dump(self.SumWeightsNeg, fout, encoding="utf-8",indent=2)
        
        tfile = self.ilpdir + str(round) + '_' + '_'.join(train_lectures) + '_t'+ '.json'
        with open(tfile, 'w') as fout:
            json.dump(self.t, fout, encoding="utf-8",indent=2)
            
    def train(self, train_lectures):
        self.stage = stage_train
        
        round, nextround = self.get_round(train_lectures)
        
        self.load_weight(train_lectures, round)
        
        for round in range(nextround, nextround+1):
            weightfile = self.ilpdir + str(round) + '_' + '_'.join(train_lectures) + '_weight_'  + "_" + '.json'
            bigramfile = ilpdir + str(round) + '_' + '_'.join(train_lectures) + '_bigram_'  + "_" + '.json'
        
            for sheet in train_lectures:
                week = int(sheet) + 1
                dir = self.ilpdir + str(week) + '/'
                
                for type in self.types:
                    self.prefix = dir + type + "." + np
                    self.featurefile = featuredir + str(week) + '/' + type + featureext
                    
                    print "update weight, round ", round
                    self.preceptron_update()
                    
            self.save_weight(train_lectures, round)
    
    def preceptron_update(self):
        #generate a system summary
        self.decode() #Line 6
        
        #update Weights, and WeightsNeg
        #read the summary, update the weight 
        sumfile = self.prefix + '.L' + str(self.L) + '.summary'
    
        #if len(fio.ReadFile(sumfile)) == 0:#no summary is generated, using a random baseline      
        #    generate_randomsummary(self.prefix, self.L, sumfile)
        
        _, System_IndexBigram, System_PhraseBigram = ILP.getPhraseBigram(sumfile, Ngram=self.ngram, MalformedFlilter=self.MalformedFlilter)
        
        #scan all the bigrams in the responses
        _, Response_IndexBigram, Response_PhraseBigram = ILP.getPhraseBigram(self.prefix+phraseext, Ngram=self.ngram, MalformedFlilter=self.MalformedFlilter)
        
        reffile = ExtractRefSummaryPrefix(self.prefix) + '.ref.summary'
        _, Model_IndexBigram, Model_PhraseBigram = ILP.getPhraseBigram(reffile, Ngram=self.ngram, MalformedFlilter=self.MalformedFlilter)
        
        Model_BigramDict = getBigramDict(Model_IndexBigram, Model_PhraseBigram)
        System_BigramDict = getBigramDict(System_IndexBigram, System_PhraseBigram)
        
        #update the weights
        FeatureVecU = LoadFeatureSet(self.featurefile)
            
        pos = 0
        neg = 0
        
        pos_bigram = []
        neg_bigram = []
        
        for summary, bigrams in Response_PhraseBigram.items():
            for bigram in bigrams:
                bigramname = Response_IndexBigram[bigram]
                if bigramname not in FeatureVecU: 
                    print bigramname
                    continue
                
                vec = FeatureVector(FeatureVecU[bigramname])
                
                my_flag = False
                if bigramname in Model_BigramDict and bigramname not in System_BigramDict:
                    print bigramname
                    
                    pos_bigram.append(bigramname)
                    
                    #self.Weights += vec
                    #self.WeightsNeg -= vec
                    
                    my_flag = True
                
                if bigramname not in Model_BigramDict and bigramname in System_BigramDict:
                    neg_bigram.append(bigramname)
                    #neg += 1
                    
                    #self.Weights -= vec
                    #self.WeightsNeg += vec
                    
                    my_flag = True
                
#                 if my_flag:
#                     self.SumWeights += self.Weights
#                     self.SumWeightsNeg += self.WeightsNeg
#                     self.t += 1
        
        #shuffer the negative
        negative_updates_index = numpy.random.permutation(len(neg_bigram))
        
        for bigran in pos_bigram:
            assert(bigramname in FeatureVecU)
            vec = FeatureVector(FeatureVecU[bigramname])
            
            self.Weights += vec
            self.WeightsNeg -= vec
            
            self.SumWeights += self.Weights
            self.SumWeightsNeg += self.WeightsNeg
            self.t += 1
        
        for i, k in enumerate(negative_updates_index):
            if i >= len(pos_bigram): continue
            bigramname = neg_bigram[k]
            
            assert(bigramname in FeatureVecU)
            vec = FeatureVector(FeatureVecU[bigramname])
            
            self.Weights -= vec
            self.WeightsNeg += vec
            
            self.SumWeights += self.Weights
            self.SumWeightsNeg += self.WeightsNeg
            self.t += 1
        
        print "pos:", len(pos_bigram)
        print "neg:", min(len(pos_bigram), len(neg_bigram))
    
    def get_weight_product(self):
        BigramWeights = {}
        
        for bigram in self.BigramPhrase:
            bigramname = self.IndexBigram[bigram]
                    
            if bigramname in self.FeatureVecU:
                fvec = FeatureVector(self.FeatureVecU[bigramname])
                
                #w = (self.Weights - self.WeightsNeg)
                #w = self.WeightsNeg.dot(fvec)
                
                if self.stage == stage_train:
                    w = self.Weights.dot(fvec)
                    
                    if self.weight_normalization == 4:
                        w_neg = self.WeightsNeg.dot(fvec)
                        w = numpy.exp(w - scipy.misc.logsumexp([w, w_neg]))
                elif self.stage == stage_test:#use averaged weight, TODO
                    #w = (self.SumWeights/self.t).dot(fvec)
                    #AveW = self.SumWeights* (1.0/self.t)
                    AveW = FeatureVector(self.SumWeights)
                    w = AveW.scaling(1.0/self.t).dot(fvec)
                    
                    if self.weight_normalization == 4:
                        AveWNeg = FeatureVector(self.SumWeightsNeg)
                        w_neg = AveWNeg.scaling(1.0/self.t).dot(fvec)
                    
                        w = numpy.exp(w - scipy.misc.logsumexp([w, w_neg]))
                else:
                    print "stage is wrong"
                    exit(-1)
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
        elif self.weight_normalization == 3:#sigmoid 
            for bigram in BigramWeights:
                w = BigramWeights[bigram]
                if w <= -100: w = -100
                BigramWeights[bigram] = 1 / (1 + math.exp(-w))
        elif self.weight_normalization == 4:#softmax
            pass
        else:
            pass
                    
        return BigramWeights

    def formulate_problem(self):
        fio.remove(self.lpfile + lpext)
        
        lines = []
        
        #write objective
        lines.append("Maximize")
        objective = []
        
        BigramWeights = self.get_weight_product()
        
#         if os.name == 'nt':
#             import matplotlib.pyplot as plt
#             plt.clf()
#             plt.hist(BigramWeights.values(), bins=50)
#             plt.savefig(self.lpfile + '.png')
        
        fio.SaveDict(BigramWeights, self.lpfile + '.bigram_weight.txt', SortbyValueflag=True)
        
        if self.student_coverage:
            for bigram in self.BigramPhrase:
                if bigram not in BigramWeights: 
                    lines.append(self.IndexBigram[bigram])
                    continue
                
                w = BigramWeights[bigram]
                
                if w <= 0: continue
                objective.append(" ".join([str(w*self.student_lambda), bigram]))
                         
            for student, grama in self.StudentGamma.items():
                if self.student_lambda==1:continue
                 
                objective.append(" ".join([str(grama*(1-self.student_lambda)), student]))
        else:
            for bigram in self.BigramPhrase:
                if bigram not in BigramWeights: continue
                bigramname = self.IndexBigram[bigram]
                        
                w = BigramWeights[bigram]
                
                if w <= 0: continue
                objective.append(" ".join([str(w), bigram]))
        
        lines.append("  " + " + ".join(objective))
        
        #write constraints
        lines.append("Subject To")
        lines += ILP.WriteConstraint1(self.PhraseBeta, self.L)
        
        lines += ILP.WriteConstraint2(self.BigramPhrase)
        
        lines += ILP.WriteConstraint3(self.PhraseBigram)
        
        if self.student_coverage:
            lines += ILP.WriteConstraint4(self.StudentPhrase)
        
        indicators = []
        for bigram in self.BigramPhrase.keys():
            indicators.append(bigram)
        for phrase in self.PhraseBeta.keys():
            indicators.append(phrase)
            
        if self.student_coverage:
            for student in self.StudentGamma.keys():
                indicators.append(student)
            
        #write Bounds
        lines.append("Bounds")
        for indicator in indicators:
            lines.append("  " + indicator + " <= " + str(1))
        
        #write Integers
        lines.append("Integers")
        lines.append("  " + " ".join(indicators))
        
        #write End
        lines.append("End")
        fio.SaveList(lines, self.lpfile + lpext)
        
    def test(self, train_lectures, test_lectures):
        self.stage = stage_test
        self.train_lectures = train_lectures
        
        round, nextround = self.get_round(train_lectures)
        self.load_weight(train_lectures, round)
            
        for sheet in test_lectures:
            week = int(sheet) + 1
            dir = ilpdir + str(week) + '/'
            
            for type in self.types:
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
    
if __name__ == '__main__':   
    
    from config import ConfigFile
    config = ConfigFile()
    
    ilpdir = "../../data/ILP_CW/"
    
    featuredir = ilpdir
    
    MalformedFlilter = False
    ngrams = config.get_ngrams()
    
    numpy.random.seed(0)
    
    #for Lambda in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
    for Lambda in [config.get_student_lambda()]:
         #for L in [10, 15, 20, 25, 30, 35, 40, 45, 50]:
         for L in [config.get_length_limit()]:
             for np in ['sentence']: #'chunk
                 ilp = ConceptWeightILP(ilpdir, np, L, ngrams, MalformedFlilter, featuredir, 
                                         student_coverage = config.get_student_coverage(), 
                                         student_lambda = config.get_student_lambda(), 
                                         minthreshold=config.get_perceptron_threshold(), 
                                         weight_normalization=config.get_weight_normalization(), no_training=config.get_no_training(), types = config.get_types())
                 for iter in range(config.get_perceptron_maxIter()):
                     ilp.run_crossvalidation()
                 ilp.gather_rouges()
    
    print "done"