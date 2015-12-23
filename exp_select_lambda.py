import random
import numpy
import fio
from scipy.stats import ttest_rel as ttest #paired t-test
import stats_util
from sklearn.cross_validation import KFold

def getRougesbyWeek(input):
    head, body = fio.ReadMatrix(input, hasHead=True)        
    return body

def get_ttest_pvalues(body1, body2, index):
    p_values = []
    for k in index:
        X = [float(row[k]) for row in body1]
        Y = [float(row[k]) for row in body2]
        _, p = stats_util.ttest(X, Y, tail=1, type=1)
        
        p_values.append(p)
    
    return p_values
    
def select_lambda_MC_Engnieer_dev(): # a random 3 lectures
    head = ['week', 'R1-R', 'R1-P', 'R1-F', 'R2-R', 'R2-P', 'R2-F', 'RSU4-R', 'RSU4-P', 'RSU4-F']
    score_index = [head.index('R1-F'), head.index('R2-F'), head.index('RSU4-F')]
    
    baseline_rouges_file =  "../../data/Engineer/ILP_Baseline/rouge.sentence.L30.txt"
    baseline_rouges = getRougesbyWeek(baseline_rouges_file)
    
    weeks = range(1, 13)
    
    selected = [2, 3, 4]
    test_set = [week for week in weeks if week not in selected]
    
    print "dev:", sorted(selected)
    print "test:", sorted(test_set)
    
    dev_scores = {}
    
    for softimpute_lambda in numpy.arange(0.5, 6.0, 0.5):
        for sparse in [0]:
            ilpdir = "../../data/Engineer/ILP_MC/"
            
            rougename = ilpdir + 'rouge.sentence.L' +str(30) + '.l' + str(softimpute_lambda) +'.s'+ str(sparse) + ".txt"
            
            scores = getRougesbyWeek(rougename)
            
            #R1
            selected_scores = []
            for week in selected:
                for row in scores:
                    if row[0] != 'average' and int(row[0]) == week:
                        selected_scores.append(float(row[6]))
    
            ave_r1 = numpy.mean(selected_scores)
            
            dev_scores[str(softimpute_lambda)] = ave_r1
    
    sorted_lambdas = sorted(dev_scores, key=dev_scores.get, reverse=True)
    
    print "lambda:", sorted_lambdas[0]
    
    softimpute_lambda = sorted_lambdas[0]
    sparse = 0
    
    #get test scores
    rougename = ilpdir + 'rouge.sentence.L' +str(30) + '.l' + str(softimpute_lambda) +'.s'+ str(sparse) + ".txt"
    scores = getRougesbyWeek(rougename)
    
    selected_baseline_scores = []
    for week in test_set:
        for row in baseline_rouges:
            if row[0] != 'average' and int(row[0]) == week:
                selected_baseline_scores.append(row)
    
    selected_test_scores = []
    for week in test_set:
        for row in scores:
            if row[0] != 'average' and int(row[0]) == week:
                selected_test_scores.append(row)
    
    pvalues = get_ttest_pvalues(selected_baseline_scores, selected_test_scores, score_index)
    
    print pvalues
    
    count = 0
    for p in pvalues:
        if p <= 0.05:
            count += 1
    return count

def select_lambda_MC_Engnieer_cv(): # a random 3 lectures
    head = ['week', 'R1-R', 'R1-P', 'R1-F', 'R2-R', 'R2-P', 'R2-F', 'RSU4-R', 'RSU4-P', 'RSU4-F']
    #score_index = [head.index('R1-F'), head.index('R2-F'), head.index('RSU4-F')]
    score_index = range(1, 10)
    
    baseline_rouges_file =  "../../data/Engineer/ILP_Baseline/rouge.sentence.L30.txt"
    baseline_rouges = getRougesbyWeek(baseline_rouges_file)
    
    weeks = [2, 6, 8, 10, 3, 7, 9, 12, 1, 4, 5, 11]
    
    kf = KFold(len(weeks), n_folds=3, shuffle=False)
    
    selected_test_scores = []
    selected_baseline_scores = []
    
    for train_index, test_index in kf:
        selected = [weeks[i] for i in train_index]
        test_set = [weeks[i] for i in test_index]
        
        print "dev:", sorted(selected)
        print "test:", sorted(test_set)
        
        dev_scores = {}
        
        for softimpute_lambda in numpy.arange(0.5, 6.0, 0.5):
            for sparse in [0]:
                ilpdir = "../../data/Engineer/ILP_MC/"
                
                rougename = ilpdir + 'rouge.sentence.L' +str(30) + '.l' + str(softimpute_lambda) +'.s'+ str(sparse) + ".txt"
                
                scores = getRougesbyWeek(rougename)
                
                #R1
                selected_scores = []
                for week in selected:
                    for row in scores:
                        if row[0] != 'average' and int(row[0]) == week:
                            selected_scores.append(float(row[3]))
        
                ave_r1 = numpy.mean(selected_scores)
                
                dev_scores[str(softimpute_lambda)] = ave_r1
        
        sorted_lambdas = sorted(dev_scores, key=dev_scores.get, reverse=True)
        
        print "lambda:", sorted_lambdas[0]
        
        softimpute_lambda = sorted_lambdas[0]
        sparse = 0
        
        #get test scores
        rougename = ilpdir + 'rouge.sentence.L' +str(30) + '.l' + str(softimpute_lambda) +'.s'+ str(sparse) + ".txt"
        scores = getRougesbyWeek(rougename)
        
        for week in test_set:
            for row in scores:
                if row[0] != 'average' and int(row[0]) == week:
                    selected_test_scores.append(row)
        
        for week in test_set:
            for row in baseline_rouges:
                if row[0] != 'average' and int(row[0]) == week:
                    selected_baseline_scores.append(row)
        
    pvalues = get_ttest_pvalues(selected_baseline_scores, selected_test_scores, score_index)
    
    print pvalues
    
    count = 0
    for p in pvalues:
        if p <= 0.05:
            count += 1
    
    fio.WriteMatrix("../../data/Engineer/test.txt", selected_test_scores, head)
    return count

if __name__ == '__main__':
    #select_lambda_MC_Engnieer_dev()
    
    select_lambda_MC_Engnieer_cv()
