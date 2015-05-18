import sys
import os
import fio

def getRouges(input):
    head, body = fio.ReadMatrix(input, hasHead=True)        
    return body[-1][1:]

def run_UnsupervisedMC():
    from config import ConfigFile
    
    config = ConfigFile()
    
    Header = ['method'] + ['R1-R', 'R1-P', 'R1-F', 'R2-R', 'R2-P', 'R2-F', 'RSU4-R', 'RSU4-P', 'RSU4-F']*3
    body = []
    
    old_rank = config.get_rank_max()
    old_sparse = config.get_sparse_threshold()
    
    for rank in [20, 50, 100, 200, 500, 1000]:
    #for rank in [100]:
        for sparse in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0]:
        #for sparse in [1.0, 0.9]:
            config.set_rank_max(rank)
            config.set_sparse_threshold(sparse)
            config.save()
            
            row = ['baseline+MC']
            row.append(rank)
            row.append(sparse)
            
            rougename = '../../data/ILP1_Sentence_MC/rouge.sentence.L' +str(config.get_length_limit())+ '.' + str(rank) +'.'+ str(sparse) + ".txt"
            
            if not fio.IsExist(rougename):
                os.system('python ILP_MC.py')
                os.system('python ILP_GetRouge.py "../../data/ILP1_Sentence_MC/"')
                
                rougefile = "../../data/ILP1_Sentence_MC/" + "rouge.sentence.L"+str(config.get_length_limit())+".txt"
                os.system('mv ' + rougefile + ' ' + rougename)
                
            scores = getRouges(rougename)
            
            row = row + scores
            body.append(row)
    
    config.set_rank_max(old_rank)
    config.set_sparse_threshold(old_sparse)
    config.save()
                
    newname = "../../data/baseline_MC_sparse_threshold.txt"
    fio.WriteMatrix(newname, body, Header)

def run_SupervisedMC():
    from config import ConfigFile
    
    config = ConfigFile()
    
    Header = ['method'] + ['R1-R', 'R1-P', 'R1-F', 'R2-R', 'R2-P', 'R2-F', 'RSU4-R', 'RSU4-P', 'RSU4-F']*3
    body = []
    
    old_rank = config.get_rank_max()
    old_sparse = config.get_sparse_threshold()
    
    for rank in [20, 50, 100, 200, 500, 1000]:
    #for rank in [100]:
        for sparse in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0]:
        #for sparse in [1.0, 0.9]:
            config.set_rank_max(rank)
            config.set_sparse_threshold(sparse)
            config.save()
            
            row = ['featureweight+MC']
            row.append(rank)
            row.append(sparse)
            
            rougename = '../../data/ILP_Sentence_Supervised_FeatureWeightingMC/rouge.sentence.L' +str(config.get_length_limit())+ '.' + str(rank) +'.'+ str(sparse) + ".txt"
            
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
                
    newname = "../../data/featureweighting_MC_sparse_threshold.txt"
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
        
if __name__ == '__main__':
    #run_UnsupervisedMC()
    run_SupervisedMC_weighting()