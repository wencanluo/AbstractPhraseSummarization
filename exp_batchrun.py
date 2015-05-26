import sys
import os
import fio

import ILP_MC
from ILP_baseline import getRouges

def get_Sparse():
    from config import ConfigFile
    
    head = ['sparse ratio', 'NoneZero']
    body = []
    config = ConfigFile()
    matrix_dir = config.get_matrix_dir()

    for eps in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        row = [eps]
        
        row.append(ILP_MC.getSparseRatio(matrix_dir, prefixA=config.get_prefixA(), eps=eps))
        body.append(row)
    
    fio.WriteMatrix(matrix_dir + 'sparse_ratio' + config.get_prefixA() + '.txt', body, head)
    
def run_Baseline():
    from config import ConfigFile
    
    config = ConfigFile()
    
    Header = ['method', 'L'] + ['R1-R', 'R1-P', 'R1-F', 'R2-R', 'R2-P', 'R2-F', 'RSU4-R', 'RSU4-P', 'RSU4-F']*3
    body = []
    
    old_length = config.get_length_limit()
    
    for L in [20, 25, 30, 35, 40]:
        config.set_length_limit(L)
        config.save()
        
        row = ['baelinse']
        row.append(L)
        
        ilpdir = "../../data/ILP1_Sentence/"
        
        rougename = ilpdir + 'rouge.sentence.L' +str(config.get_length_limit())+ ".txt"
        
        if not fio.IsExist(rougename):
            os.system('python ILP_baseline.py')
            os.system('python ILP_GetRouge.py ' + ilpdir)
            
        scores = getRouges(rougename)
        
        row = row + scores
        body.append(row)
    
    config.set_length_limit(old_length)
    config.save()
                
    newname = ilpdir + "rouge.sentence.txt"
    fio.WriteMatrix(newname, body, Header)
    
def run_UnsupervisedMC():
    from config import ConfigFile
    
    config = ConfigFile()
    
    Header = ['method'] + ['R1-R', 'R1-P', 'R1-F', 'R2-R', 'R2-P', 'R2-F', 'RSU4-R', 'RSU4-P', 'RSU4-F']*3
    body = []
    
    old_length = config.get_length_limit()
    old_sparse = config.get_sparse_threshold()
    
    for L in [20, 25, 30, 35, 40]:
        config.set_length_limit(L)
        for sparse in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0]:
            config.set_sparse_threshold(sparse)
            config.save()
            
            row = ['baseline+MC']
            row.append(L)
            row.append(sparse)
            
            ilpdir = "../../data/ILP1_Sentence_MC/"
            
            rougename = ilpdir + 'rouge.sentence.L' +str(config.get_length_limit()) +'.'+ str(sparse) + ".txt"
            
            if not fio.IsExist(rougename):
                os.system('python ILP_MC.py')
                os.system('python ILP_GetRouge.py ' + ilpdir)
                
                rougefile = ilpdir + "rouge.sentence.L"+str(config.get_length_limit())+".txt"
                os.system('mv ' + rougefile + ' ' + rougename)
                
            scores = getRouges(rougename)
            
            row = row + scores
            body.append(row)
    
    config.set_length_limit(old_length)
    config.set_sparse_threshold(old_sparse)
    config.save()
                
    newname = ilpdir + "rouge.sentence.txt"
    fio.WriteMatrix(newname, body, Header)

def run_SupervisedMC():
    from config import ConfigFile
    
    config = ConfigFile()
    
    Header = ['method'] + ['R1-R', 'R1-P', 'R1-F', 'R2-R', 'R2-P', 'R2-F', 'RSU4-R', 'RSU4-P', 'RSU4-F']*3
    body = []
    
    old_rank = config.get_rank_max()
    old_sparse = config.get_sparse_threshold()
    
    ilpdir = "../../data/ILP_Sentence_Supervised_FeatureWeightingAveragePerceptronMC/"
    
    for sparse in [0, 0.9, 0.8, 0.1]:
        for rank in [20, 50, 100, 200, 500]:
            os.system('rm ' + ilpdir + '*.json')
            
            config.set_rank_max(rank)
            config.set_sparse_threshold(sparse)
            config.save()
            
            row = ['featureweight+MC']
            row.append(rank)
            row.append(sparse)
            
            rougename = ilpdir + 'rouge.sentence.L' +str(config.get_length_limit())+ '.w' + str(config.get_weight_normalization()) + config.get_prefixA() +'.s'+ str(sparse) + ".txt"
            
            if not fio.IsExist(rougename):
                os.system('python ILP_Supervised_FeatureWeight_MC_AveragePerceptron.py')
                os.system('python ILP_GetRouge.py ' + ilpdir)
                
                rougefile = ilpdir + "rouge.sentence.L"+str(config.get_length_limit())+".txt"
                os.system('mv ' + rougefile + ' ' + rougename)
                
            scores = getRouges(rougename)
            
            row = row + scores
            body.append(row)
    
    config.set_rank_max(old_rank)
    config.set_sparse_threshold(old_sparse)
    config.save()
                
    newname = ilpdir + 'rouge.sentence.L' +str(config.get_length_limit())+ '.w' + str(config.get_weight_normalization()) + '.txt'
    fio.WriteMatrix(newname, body, Header)

def run_SupervisedMC2():
    from config import ConfigFile
    
    config = ConfigFile()
    
    Header = ['method'] + ['R1-R', 'R1-P', 'R1-F', 'R2-R', 'R2-P', 'R2-F', 'RSU4-R', 'RSU4-P', 'RSU4-F']*3
    body = []
    
    old_rank = config.get_rank_max()
    old_sparse = config.get_sparse_threshold()
    
    ilpdir = "../../data/ILP_Sentence_Supervised_FeatureWeightingAveragePerceptronMC/"
    
    old_length = config.get_length_limit()
    
    for L in [20, 25, 30, 35, 40]:
        config.set_length_limit(L)
        for sparse in [0, 0.9]:
            for rank in [1000]:
                os.system('rm ' + ilpdir + '*.json')
                
                config.set_rank_max(rank)
                config.set_sparse_threshold(sparse)
                config.save()
                
                row = ['featureweight+MC']
                row.append(rank)
                row.append(sparse)
                
                rougename = ilpdir + 'rouge.sentence.L' +str(config.get_length_limit())+ '.w' + str(config.get_weight_normalization()) + config.get_prefixA() +'.s'+ str(sparse) + ".txt"
                
                if not fio.IsExist(rougename):
                    os.system('python ILP_Supervised_FeatureWeight_MC_AveragePerceptron.py')
                    os.system('python ILP_GetRouge.py ' + ilpdir)
                    
                    rougefile = ilpdir + "rouge.sentence.L"+str(config.get_length_limit())+".txt"
                    os.system('mv ' + rougefile + ' ' + rougename)
                    
                scores = getRouges(rougename)
                
                row = row + scores
                body.append(row)
    
    config.set_length_limit(old_length)
    config.set_rank_max(old_rank)
    config.set_sparse_threshold(old_sparse)
    config.save()
                
    newname = ilpdir + 'rouge.sentence' + '.w' + str(config.get_weight_normalization()) + '.txt'
    fio.WriteMatrix(newname, body, Header)
    
def run_SupervisedMC_weighting():
    from config import ConfigFile
    
    config = ConfigFile()
    
    Header = ['method'] + ['R1-R', 'R1-P', 'R1-F', 'R2-R', 'R2-P', 'R2-F', 'RSU4-R', 'RSU4-P', 'RSU4-F']*3
    body = []
    
    old_rank = config.get_rank_max()
    old_sparse = config.get_sparse_threshold()
    old_threshold = config.get_perceptron_threshold()
    
    for threshold in [-500, -400, -300, -200, -150, -100, -80, -50, 0]:
        config.set_perceptron_threshold(threshold)
        config.save()
            
        row = ['featureweight+MC']
        row.append(threshold)
        
        rougename = '../../data/ILP_Sentence_Supervised_FeatureWeightingMC/rouge.sentence.L' +str(config.get_length_limit())+ '.' + str(threshold) + ".txt"
        
        if not fio.IsExist(rougename):
            os.system('python ILP_Supervised_FeatureWeight_MC.py')
            os.system('python ILP_GetRouge.py "../../data/ILP_Sentence_Supervised_FeatureWeightingMC/"')
            
            rougefile = "../../data/ILP_Sentence_Supervised_FeatureWeightingMC/" + "rouge.sentence.L"+str(config.get_length_limit())+".txt"
            os.system('mv ' + rougefile + ' ' + rougename)
            
        scores = getRouges(rougename)
        
        row = row + scores
        body.append(row)
    
    config.set_rank_max(old_rank)
    config.set_sparse_threshold(old_sparse)
    config.save()
                
    newname = "../../data/featureweighting_MC_perceptron_threshold.txt"
    fio.WriteMatrix(newname, body, Header)

def run_CWLearning():
    from config import ConfigFile
    
    config = ConfigFile()
    old_weight = config.get_weight_normalization()
    old_maxiter = config.get_perceptron_maxIter()
    
    config.set_weight_normalization(3)
    config.set_perceptron_maxIter(10)
    config.save()
    
    rougename = '../../data/ILP_Sentence_Supervised_FeatureWeightingAveragePerceptron/rouge.sentence.L' +str(config.get_length_limit())+ '.' + str(config.get_weight_normalization()) + ".txt"
    
    if not fio.IsExist(rougename):
        os.system('python ILP_Supervised_FeatureWeight_AveragePerceptron.py')
        os.system('python ILP_GetRouge.py "../../data/ILP_Sentence_Supervised_FeatureWeightingMC/"')
        
        rougefile = "../../data/ILP_Sentence_Supervised_FeatureWeightingMC/" + "rouge.sentence.L"+str(config.get_length_limit())+".txt"
        os.system('mv ' + rougefile + ' ' + rougename)
    
    config.set_weight_normalization(old_weight)
    config.set_perceptron_maxIter(old_maxiter)
    config.save()
     
if __name__ == '__main__':
    #run_Baseline()
    #run_UnsupervisedMC()
    #get_Sparse()
    run_SupervisedMC()
    run_SupervisedMC2()