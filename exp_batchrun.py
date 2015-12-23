import sys
import os
import fio

import numpy

import ILP_MC
from ILP_baseline import getRouges

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
    
    Header = ['method', 'L', 'lambda', 'sparse'] + ['R1-R', 'R1-P', 'R1-F', 'R2-R', 'R2-P', 'R2-F', 'RSU4-R', 'RSU4-P', 'RSU4-F']*3
    body = []
    
    old_length = config.get_length_limit()
    old_sparse = config.get_sparse_threshold()
    old_softimpute_lambda = config.get_softImpute_lambda()
    
    #for L in [20, 25, 30, 35, 40]:
    for L in [30]:
        config.set_length_limit(L)
        for softimpute_lambda in numpy.arange(0.5, 8.0, 0.5):
        #for softimpute_lambda in [2.0]:
            #for sparse in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0]:
            for sparse in [0]:
                config.set_sparse_threshold(sparse)
                config.set_softImpute_lambda(softimpute_lambda)
                config.save()
                
                row = ['baseline+MC']
                row.append(L)
                row.append(softimpute_lambda)
                row.append(sparse)
                
                ilpdir = "../../data/ILP1_Sentence_MC/"
                
                rougename = ilpdir + 'rouge.sentence.L' +str(config.get_length_limit()) + '.l' + str(softimpute_lambda) +'.s'+ str(sparse) + ".txt"
                
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
    config.set_softImpute_lambda(old_softimpute_lambda)
    config.save()
                
    newname = ilpdir + "rouge.sentence.txt"
    fio.WriteMatrix(newname, body, Header)

def run_UnsupervisedMC_IE256():
    from config import ConfigFile
    
    config = ConfigFile(config_file_name='config_IE256.txt')
    
    Header = ['method', 'L', 'lambda', 'sparse'] + ['R1-R', 'R1-P', 'R1-F', 'R2-R', 'R2-P', 'R2-F', 'RSU4-R', 'RSU4-P', 'RSU4-F']*3
    
    
    old_length = config.get_length_limit()
    old_sparse = config.get_sparse_threshold()
    old_softimpute_lambda = config.get_softImpute_lambda()
    
    for L in [40, 39]:
    #for L in [27, 33]:
    #for L in [15, 16]:
    #for L in [39]:
        body = []
        
        config.set_length_limit(L)
        for softimpute_lambda in ['org'] + list(numpy.arange(0.0, 5.1, 0.1)):
            if softimpute_lambda == 'org':
                sparses = [1.0]
            else:
                #sparses = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
                sparses = [0.0]
                
            for sparse in sparses:
                config.set_sparse_threshold(sparse)
                config.set_softImpute_lambda(softimpute_lambda)
                config.save()
                
                row = ['baseline+MC']
                row.append(L)
                row.append(softimpute_lambda)
                row.append(sparse)
                
                ilpdir = "../../data/IE256/ILP_Sentence_MC/"
                
                rougename = ilpdir + 'rouge.sentence.L' + str(L) + '.' + str(softimpute_lambda) +'.'+ str(sparse) + ".txt"
                
                if not fio.IsExist(rougename):
                    os.system('python ILP_MC.py')
                    os.system('python IE256_ILP_GetRouge.py %s %s %s'%(ilpdir, str(softimpute_lambda), str(sparse)))
                    
                    rougefile = ilpdir + "rouge.sentence.L"+ str(L) +".txt"
                    os.system('mv ' + rougefile + ' ' + rougename)
                    
                scores = getRouges(rougename)
                
                row = row + scores
                body.append(row)
                
        newname = ilpdir + 'rouge.sentence.L' +str(L) + '.txt'
        fio.WriteMatrix(newname, body, Header)
    
    config.set_length_limit(old_length)
    config.set_sparse_threshold(old_sparse)
    config.set_softImpute_lambda(old_softimpute_lambda)
    config.save()

def run_UnsupervisedMC_DUC():
    from config import ConfigFile
    
    config = ConfigFile(config_file_name='duc_config.txt')
    
    Header = ['method', 'L', 'lambda', 'sparse'] + ['R1-R', 'R1-P', 'R1-F', 'R2-R', 'R2-P', 'R2-F', 'RSU4-R', 'RSU4-P', 'RSU4-F']*3
    body = []
    
    old_length = config.get_length_limit()
    old_sparse = config.get_sparse_threshold()
    old_softimpute_lambda = config.get_softImpute_lambda()
    
    #for L in [20, 25, 30, 35, 40]:
    for L in [100]:
        config.set_length_limit(L)
        softimpute_lambdas = ['org', '6', '5.5', '5', '4.5', '4', '3.5', '3', '2.5', '2', '1.5', '1', '0.5']
        for softimpute_lambda in softimpute_lambdas:
        #for softimpute_lambda in [2.0]:
            if softimpute_lambda == 'org':
                sparses = [1.0]
            else:
                sparses = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
                
            for sparse in sparses:
            #for sparse in [0]:
                config.set_sparse_threshold(sparse)
                config.set_softImpute_lambda(softimpute_lambda)
                config.save()
                
                row = ['baseline+MC']
                row.append(L)
                row.append(softimpute_lambda)
                row.append(sparse)
                
                ilpdir = "../../data/DUC_ILP_MC/"
                
                rougename = ilpdir + 'average_rouge.L' +str(config.get_length_limit()) + '.' + str(softimpute_lambda) +'.'+ str(sparse) + ".txt"
                print rougename
                
                if not fio.IsExist(rougename):
                    os.system('python DUC_ILP_MC.py')
                    os.system('python DUC_ILP_GetRouge.py %s %s %s' %(ilpdir, softimpute_lambda, str(sparse)) )
                                    
                scores = getRouges(rougename)
                
                row = row + scores
                body.append(row)
    
    config.set_length_limit(old_length)
    config.set_sparse_threshold(old_sparse)
    config.set_softImpute_lambda(old_softimpute_lambda)
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

def run_SupervisedMC3(iter=1):
    from config import ConfigFile
    
    config = ConfigFile()
    
    Header = ['method', 'L', 'lambda', 'sparse'] + ['R1-R', 'R1-P', 'R1-F', 'R2-R', 'R2-P', 'R2-F', 'RSU4-R', 'RSU4-P', 'RSU4-F']*3
    body = []
    
    old_softimpute_lambda = config.get_softImpute_lambda()
    old_sparse = config.get_sparse_threshold()
    old_length = config.get_length_limit()
    
    ilpdir = "../../data/ILP_Sentence_Supervised_FeatureWeightingAveragePerceptronMC/"
    
    if fio.IsExist(ilpdir+"rouge.sentence.txt"):
        path = ilpdir + "R" + str(iter) + '/'
        if not fio.IsExistPath(path):
            fio.NewPath(path)
        
        cmd = 'mv ' + ilpdir + '*.txt ' + path
        os.system(cmd)
        
    for L in [20, 25, 30, 35, 40]:
        config.set_length_limit(L)
        
        for sparse in [0]:
            #for softimpute_lambda in numpy.arange(0.1, 4.1, 0.1):
            #for softimpute_lambda in numpy.arange(0.5, 4.1, 0.5):
            for softimpute_lambda in [2.0]:
                os.system('rm ' + ilpdir + '*.json')
                
                config.set_softImpute_lambda(softimpute_lambda)
                config.set_sparse_threshold(sparse)
                config.save()
                
                row = ['featureweight+MC']
                row.append(L)
                row.append(softimpute_lambda)
                row.append(sparse)
                
                rougename = ilpdir + 'rouge.sentence.L' +str(config.get_length_limit()) + '.l' + str(softimpute_lambda) +'.s'+ str(sparse) + ".txt"
                
                if not fio.IsExist(rougename):
                    os.system('python ILP_Supervised_FeatureWeight_MC_AveragePerceptron.py')
                    os.system('python ILP_GetRouge.py ' + ilpdir)
                    
                    rougefile = ilpdir + "rouge.sentence.L"+str(config.get_length_limit())+".txt"
                    os.system('mv ' + rougefile + ' ' + rougename)
                    
                scores = getRouges(rougename)
                
                row = row + scores
                body.append(row)
    
    config.set_length_limit(old_length)
    config.set_softImpute_lambda(old_softimpute_lambda)
    config.set_sparse_threshold(old_sparse)
    config.save()
                
    newname = ilpdir + "rouge.sentence.txt"
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

def run_CWLearning(iter = 1):
    from config import ConfigFile
    
    config = ConfigFile()
    
    Header = ['method', 'L'] + ['R1-R', 'R1-P', 'R1-F', 'R2-R', 'R2-P', 'R2-F', 'RSU4-R', 'RSU4-P', 'RSU4-F']*3
    body = []
    
    old_length = config.get_length_limit()
    old_softimpute_lambda = config.get_softImpute_lambda()
    
    ilpdir = "../../data/ILP_Sentence_Supervised_FeatureWeightingAveragePerceptron/"
    
    if iter != None:
        if fio.IsExist(ilpdir+"rouge.sentence.L30.txt"):
            path = ilpdir + "R" + str(iter) + '/'
            if not fio.IsExistPath(path):
                fio.NewPath(path)
            
            cmd = 'mv ' + ilpdir + '*.txt ' + path
            os.system(cmd)
    
    for L in [20, 25, 30, 35, 40]:
        config.set_length_limit(L)
        config.save()
        
        #assert(config.get_softImpute_lambda() == 2.0)
        
        for softimpute_lambda in numpy.arange(0.5, 4.1, 0.5): 
            row = ['baelinse + CW']
            row.append(L)
            row.append(softimpute_lambda)
    
            rougename = ilpdir + 'rouge.sentence.L' +str(config.get_length_limit()) + '.l' + str(softimpute_lambda) + ".txt"
            
            if not fio.IsExist(rougename):
                os.system('rm ' + ilpdir + '*.json')
                
                os.system('python ILP_Supervised_FeatureWeight_AveragePerceptron.py')
                os.system('python ILP_GetRouge.py ' + ilpdir)
                
                rougefile = ilpdir + "rouge.sentence.L"+str(config.get_length_limit())+".txt"
                os.system('mv ' + rougefile + ' ' + rougename)
            
            scores = getRouges(rougename)
            
            row = row + scores
            body.append(row)
    
    newname = ilpdir + "rouge.sentence.txt"
    fio.WriteMatrix(newname, body, Header)
    
    config.set_softImpute_lambda(old_softimpute_lambda)
    config.set_length_limit(old_length)
    config.save()

def getA(rougefile):
    head, body = fio.ReadMatrix(rougefile, hasHead=True)
    
    RougeHead = ['R1-R', 'R1-P', 'R1-F', 'R2-R', 'R2-P', 'R2-F', 'RSU4-R', 'RSU4-P', 'RSU4-F']
    
    A = []
    for i, row in enumerate(body):
        if i == len(body) - 1: continue
        
        N = len(row)
        score_q3 = [float(x) for x in row[N - len(RougeHead):]]
        score_q2 = [float(x) for x in row[N - len(RougeHead)*2:N - len(RougeHead)]]
        score_q1 = [float(x) for x in row[N - len(RougeHead)*3:N - len(RougeHead)*2]]
        
        for i, score in enumerate([score_q1, score_q2, score_q3]):
            A.append(score)
    return A
    
def get_average_CWLearning():
    ilpdir = "../../data/ILP_Sentence_Supervised_FeatureWeightingAveragePerceptronMC/"
    import numpy as np
    
    rougefile = '../../data/ILP1_Sentence_MC/rouge.sentence.L30.l2.0.s0.txt'
    A = getA(rougefile)        
    fio.WriteMatrix(rougefile + '.matrix', A, header=None)
        
    SumA = []
    for r in ['R1', 'R2', 'R3', 'R4', 'R5']:
        rougefile = ilpdir + r + "/rouge.sentence.L30.l2.0.s0.txt"

        A = getA(rougefile)
            
        fio.WriteMatrix(rougefile + '.matrix', A, header=None)
        
        A = np.array(A)
        
        if SumA == []:
            SumA = A
        else:
            SumA += A
    
    fio.WriteMatrix(ilpdir + "/rouge.sentence.L30.average.txt", A, header=None)
    
    
if __name__ == '__main__':
    #get_average_CWLearning()
    
    #run_Baseline()
    run_UnsupervisedMC_IE256()
    #run_UnsupervisedMC_DUC()
#     for iter in [5]:
#         run_CWLearning(iter)
    #get_Sparse()
#     for iter in [2, 3, 4, 5]:
#         run_SupervisedMC3(iter)
    #run_SupervisedMC2()
    