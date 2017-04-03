import random
import numpy
import fio
from scipy.stats import ttest_rel as ttest #paired t-test
import stats_util
from sklearn.cross_validation import KFold
import global_params
from _collections import defaultdict
import os

def getRougesbyWeek(input):
    head, body = fio.ReadMatrix(input, hasHead=True)        
    return body

def get_ttest_pvalues(body1, body2, index, type=1):
    p_values = []
    for k in index:
        X = [float(row[k]) for row in body1]
        Y = [float(row[k]) for row in body2]
        _, p = stats_util.ttest(X, Y, tail=2, type=type)
        
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

def select_lambda_MC_cv(cid, L, nfolds): # a random 3 lectures
    sheets = global_params.lectures[cid]
    
    head = ['week', 'R1-R', 'R1-P', 'R1-F', 'R2-R', 'R2-P', 'R2-F', 'RSU4-R', 'RSU4-P', 'RSU4-F']
    metric = head.index('R1-F')
    
    #score_index = [head.index('R1-F'), head.index('R2-F'), head.index('RSU4-F')]
    score_index = range(1, 10)
    
    ilpdir = "../../data/%s/ILP_MC/"%cid
    
    if cid.startswith('Engineer'):
        baseline_rouges_file =  "../../data/%s/ILP_Baseline/rouge.sentence.L%d.None.txt"%(cid, L)
    else:
        baseline_rouges_file =  ilpdir + 'rouge.sentence.L' +str(L) + '.' + str(0.0) +'.'+ str(0.0) + ".txt"
    baseline_rouges = getRougesbyWeek(baseline_rouges_file)
    
    weeks = sheets
    
    #shuffle the weeks ?
    
    kf = KFold(len(weeks), n_folds=nfolds, shuffle=False)
    
    selected_test_scores = []
    selected_baseline_scores = []
    
    lamda_dict = defaultdict(list)
    
    for train_index, test_index in kf:
        selected = [weeks[i] for i in train_index]
        test_set = [weeks[i] for i in test_index]
        
        #print "dev:", sorted(selected)
        #print "test:", sorted(test_set)
        
        dev_scores = {}
        
        for softimpute_lambda in numpy.arange(0.5, 5.5, 0.5):
            for sparse in [0.0]:
                
                rougename = ilpdir + 'rouge.sentence.L' +str(L) + '.' + str(softimpute_lambda) +'.'+ str(sparse) + ".txt"
                
                scores = getRougesbyWeek(rougename)
                
                #R1
                selected_scores = []
                for week in selected:
                    for row in scores:
                        if not row[0].startswith('ave') and int(row[0]) == week:
                            selected_scores.append(float(row[metric]))
        
                ave_r1 = numpy.mean(selected_scores)
                
                dev_scores[str(softimpute_lambda)] = ave_r1
        
        sorted_lambdas = sorted(dev_scores, key=dev_scores.get, reverse=True)
        
        #print "lambda:", sorted_lambdas[0]
        
        softimpute_lambda = sorted_lambdas[0]
        
        for week in test_set:
            lamda_dict[week].append(softimpute_lambda)
            
        sparse = 0.0
        
        #get test scores
        rougename = ilpdir + 'rouge.sentence.L' +str(L) + '.' + str(softimpute_lambda) +'.'+ str(sparse) + ".txt"
        scores = getRougesbyWeek(rougename)
        
        for week in test_set:
            for row in scores:
                if not row[0].startswith('ave') and int(row[0]) == week:
                    selected_test_scores.append(row)
        
        for week in test_set:
            for row in baseline_rouges:
                if not row[0].startswith('ave') and int(row[0]) == week:
                    selected_baseline_scores.append(row)
        
    pvalues = get_ttest_pvalues(selected_baseline_scores, selected_test_scores, score_index)
    
    
    lamda_dict_file = os.path.join("../../data/%s/MC/"%cid, 'lambda_%d.json'%L)
    fio.SaveDict2Json(lamda_dict, lamda_dict_file)
    
    #print pvalues
    
    count = 0
#     for p in pvalues:
#         if p <= 0.05:
#             count += 1
    
    baseline_ave = ['ILP']
    row = ['ILP+MC']
    for i, p in zip(range(1, len(head)), pvalues):
        scores = [float(xx[i]) for xx in selected_test_scores]
        base_scores = [float(xx[i]) for xx in selected_baseline_scores]
        baseline_ave.append('%.3f'%(numpy.mean(base_scores)))
        
        ave = numpy.mean(scores)
        
        if ave > float(baseline_ave[-1]):
            count += 1
        
        if p < 0.05:
            row.append('%.3f%s'%(ave, '+' if ave > float(baseline_ave[-1])  else '-'))
        else:
            row.append('%.3f'%(ave))
    
    selected_test_scores.append(baseline_ave)
    selected_test_scores.append(row)
    
    fio.WriteMatrix("../../data/%s/test_%d.txt"%(cid,L), selected_test_scores, head)
    return count

if __name__ == '__main__':
    #select_lambda_MC_Engnieer_dev()
    
    #select_lambda_MC_Engnieer_cv()
    
    #cid = 'review_camera'
#     cid = 'review_IMDB'
    #cid = 'review_prHistory'
                
    #cid = 'CS0445'
    #cid = 'IE256_2016'
    #cid = 'IE256'
    
    for cid in [
#				 'Engineer', 'Engineer_nocutoff',
#                 'IE256', 'IE256_nocutoff',
#                 'IE256_2016','IE256_2016_nocutoff',
#                 'CS0445', 
#                 'CS0445_nocutoff',
#                 'review_camera', 
#                 'review_IMDB', 
#                 'review_prHistory',
                 'review_camera_nocutoff', 
                 'review_IMDB_nocutoff', 
                 'review_prHistory_nocutoff',
                #'DUC04',
                #'DUC04_nocutoff'
#                'DUC04_12.6', 'DUC04_13.9', 'DUC04_16.5', 'DUC04_17.2',
#                 'IE256_21.0',
#                 'IE256_26.5',
#                 'IE256_2016_23.7',
#                 'IE256_2016_29.9',
#                 'IE256_2016_31.2',
#                 'CS0445_28.0', 'CS0445_32.7', 'CS0445_34.2',

#				 'Engineer_16.0', 'Engineer_26.5', 
#                 'Engineer_36.0', 'Engineer_38.6',  'Engineer_41.4',
                 
#                 'review_camera_84.9', 'review_camera_85.8', 'review_camera_86.2', 
#                 'review_IMDB_76.5', 'review_IMDB_76.8', 
#                 'review_prHistory_77.4', 'review_prHistory_78.7', 'review_prHistory_80.4',
#                 #'DUC04_23.4', 'DUC04_21.2',
#                 'CS0445_11.0', 'CS0445_19.3',
#                 'IE256_5.6', 'IE256_11.9', 
#                 'IE256_2016_5.4', 
#                 'IE256_2016_13.2', 
#                 'review_camera_74.5', 'review_camera_78.7', 'review_camera_83.2', 
#                 'review_IMDB_70.8', 'review_IMDB_71.9', 'review_IMDB_74.8',
#                 'review_prHistory_71.3', 'review_prHistory_75.6',
                ]:
    
        if cid.startswith('CS0445'):
            LL = [16]
        elif cid.startswith('IE256_2016'):
            LL = [13]
        elif cid.startswith('IE256'):
            LL = [15]
        elif cid.startswith('Engineer'):
            LL = [30]
        elif cid.startswith('review_camera'):
            LL = [216]
        elif cid.startswith('review_IMDB'):
            LL = [242]
        elif cid.startswith('review_prHistory'):
            LL = [190]
        elif cid.startswith('DUC'):
            LL = [105]
        else: #news
            LL = [100]
            
        for L in LL:
            counts = []
            
            #folds = range(2, 20)
#             if cid.startswith('review'):
#                 FF = [3]
#             else:
#                 FF = [10]
            
            #FF = [3]
            FF = [len(global_params.lectures[cid])]
            #FF = [10]
			
            folds = FF
            for fold in folds:
                count = select_lambda_MC_cv(cid, L, fold)
                counts.append(count)
            print cid, L, max(counts), folds[counts.index(max(counts))], counts
            #select_lambda_MC_cv('CS0445', L)
