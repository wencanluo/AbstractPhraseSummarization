import fio
import numpy
import json
import random
import re
import codecs
import Cosin
from collections import defaultdict
import stats_util
from nltk.metrics.agreement import AnnotationTask
import numpy as np
import global_params
import os

folders = {'Mead':'../../data/Engineer/Mead/',
           'Oracle_ILP':'../../data/Engineer/ILP_Oracle/',
           'Oracle':'../../data/Engineer/Oracle/',
           'ILP':'../../data/Engineer/ILP_Baseline/',
           'MC':'../../data/Engineer/ILP_MC/',
           }

prompts = {'MP':"Describe what was confusing or needed more detail.",
           'POI':"Describe what you found most interesting in today's class.",
           'LP':"Describe what you learned about how you learn.",
           'q1':"Describe what was confusing or needed more detail.",
           'q2':"Describe what you found most interesting in today's class.",
           'q3':"Describe what you learned about how you learn.",
           }

def load_summary(input):#re-order it by cosine
    '''
    to lower,
    '''
    lines = []
    with codecs.open(input, 'r', 'utf-8') as fin:
        for line in fin:
            lines.append(line)
    
    summaries = []
    for line in lines:
        line = line.strip().lower()
        
        line = line.replace('"','\'')
        g = re.search('^\[\d+\](.*)$', line)
        if g != None:
            line = g.group(1).strip()
        
        summaries.append(line)
    return summaries

def sort_by_cosin(A, H):
    s = []
    for i, line in enumerate(A):
        max = -100
        for lineH in H:
            x = Cosin.compare(line, lineH)
            if x >= max:
                max = x
        s.append(max)
    
    sort_index = sorted(range(len(s)), key=lambda k: s[k], reverse = True)
    
    A = [A[i] for i in sort_index]
    
    return A

def list2paragraph(summaries):
    '''
    add number
    add <p></p>
    format into one line
    '''
    newLines = []
    for i, line in enumerate(summaries):
        newLines.append('<p>[%d] %s</p>'%(i+1, line))
    
    if len(newLines) == 0:
        return '"<p></p>"'
    return '"%s"'% ('\n'.join(newLines))

def generate_checking_point(cid, sheets, prompt, sheet, summaryA, summaryB, N, L):
    
    w = random.randint(0, len(sheets)-1)
    while sheets[w] == sheet:
        w = random.randint(0, len(sheets)-1)
    w = sheets[w]
    
    rp = random.random()
    if rp>0.5:#switch
        summaryB = summaryA
    else:
        summaryA = summaryB
    
    #read TA's summmary
    Hs = load_human_summary(cid, sheet, prompt, N)
                
    #read system summary
    A = load_system_sumary(cid, summaryA, L, sheet, prompt, N)

    B = load_system_sumary(cid, summaryB, L, sheet, prompt, N)
    
    p = numpy.random.permutation(len(B))
    
    B = [B[i] for i in p]
    
    logrow = [prompt, w, summaryA, summaryB, 'Y']
    row = ['"%s"'%prompts[prompt], list2paragraph(Hs[0]), list2paragraph(A), list2paragraph(B)]
                    
    return logrow, row

def get_kappa(input):
    head,body = fio.ReadMatrix(input, True)
    
    data = []
    for i,row in enumerate(body):
        for coder, label in enumerate(row):
            if label == 'a': label = '0'
            data.append((head[coder], i, label))
    
    task = AnnotationTask(data)
    
    print head[0], head[1], task.kappa_pairwise(head[0], head[1])
    print head[0], head[2], task.kappa_pairwise(head[0], head[2])
    print head[1], head[2], task.kappa_pairwise(head[1], head[2])
    return task.kappa()

def get_majority(A):
    dict = defaultdict(int)
    
    for x in A:
        dict[x] += 1
    
    keys = sorted(dict, key=dict.get, reverse=True)
    
    return keys[0]
    
def get_agreement(logfile, results, output):
    head, body = fio.ReadMatrix(logfile, hasHead=True)
    head_res, body_res = fio.ReadMatrix(results, hasHead=True)
    
    preference_index = head_res.index('Preference')
    
    subjects = []
    dict_preference = {}
    for row in body_res:
        id = row[head_res.index('id')]
        preference = int(row[preference_index])
        subject_id = row[head_res.index('Worker ID')]
        
        subjects.append(subject_id)
        
        if id not in dict_preference:
            dict_preference[id] = {}
        
        dict_preference[id][subject_id] = preference
    
    bad_annotators = []
    for row in body:
        id, prompt, week, system_A, system_B, checking_point = row
        if checking_point != 'Y': continue
        
        for subject_id, preference in dict_preference[id].items():
            preference = int(preference)
            
            if abs(preference) == 2:
                bad_annotators.append(subject_id)
    
    bad_annotators = list(set(bad_annotators)) 
    #print bad_annotators
    
    count = 0
    total_count = 0
    
    data = []
    for row in body:
        id, prompt, week, system_A, system_B, checking_point = row
        if checking_point == 'Y': continue
        
        #for subject_id, preference in dict_preference[id].items():
        values = dict_preference[id].values()
        for i in range(len(values)):
            preference = int(values[i])
            
#             if preference < 0:
#                 preference = -1
#             elif preference > 0:
#                 preference = 1
            
            values[i] = preference
            
        majority = get_majority(values)
        
        for k, preference in enumerate(values):
            #if subject_id in bad_annotators: continue
        
            if preference == majority:
                count += 1
            
            total_count += 1
    
    print count, '\t', total_count, '\t', count*1.0/total_count
               
           
def get_agreement2(logfile, results, output):
    head, body = fio.ReadMatrix(logfile, hasHead=True)
    head_res, body_res = fio.ReadMatrix(results, hasHead=True)
    
    preference_index = head_res.index('Preference')
    
    subjects = []
    dict_preference = {}
    for row in body_res:
        id = row[head_res.index('id')]
        preference = int(row[preference_index])
        subject_id = row[head_res.index('Worker ID')]
        
        subjects.append(subject_id)
        
        if id not in dict_preference:
            dict_preference[id] = {}
        
        dict_preference[id][subject_id] = preference
    
    subjects = list(set(subjects))
    
    #print dict_preference
    new_body = []
    dict = defaultdict(int)
    for row in body:
        id, prompt, week, system_A, system_B, checking_point = row
        if checking_point == 'Y': continue
        
        new_now = []
        
        for preference in dict_preference[id].values():
            preference = int(preference)
            if preference < 0:
                preference = -1
            elif preference > 0:
                preference = 1
                
            new_now.append(preference)
#             if subject not in dict_preference[id]: 
#                 preference = 5
#             else:
#                 preference = dict_preference[id][subject]
            
        
        new_body.append(new_now)
    
    fio.WriteMatrix(output, new_body, header=['1', '2', '3', '4', '5'])
    
        
def result_analyze(logfile, results, output):
    head, body = fio.ReadMatrix(logfile, hasHead=True)
    head_res, body_res = fio.ReadMatrix(results, hasHead=True)
    
    preference_index = head_res.index('Preference')
    
    dict_preference = {}
    for row in body_res:
        id = row[head_res.index('id')]
        preference = int(row[preference_index])
        subject_id = row[head_res.index('Worker ID')]
        
        if id not in dict_preference:
            dict_preference[id] = {}
        
        dict_preference[id][subject_id] = preference
    
    count = 0
    dict = defaultdict(int)
    
    #print dict_preference
    for row in body:
        id, prompt, week, system_A, system_B, checking_point = row
        if checking_point == 'Y': continue
        
        for subject, preference in dict_preference[id].items():
            if preference > 0:
                dict[system_B] += 1
            elif preference < 0:
                dict[system_A] += 1
            else:
                dict['no_preference'] += 1
        
            count += 1
    
    #print dict_preference
    perference_body = []
    for row in body:
        id, prompt, week, system_A, system_B, checking_point = row
        if checking_point == 'Y': continue
        
        row = [id, prompt, week, system_A, system_B]
        for subject, preference in dict_preference[id].items():
            row.append(preference)
        row.append(numpy.sum(dict_preference[id].values()))
        
        perference_body.append(row)
    
    fio.WriteMatrix(output, perference_body, header=None)
    print dict
    
    PerferenceValue = {}

    count = 0
    #print dict_preference
    for row in body:
        id, prompt, week, system_A, system_B, checking_point = row
        if checking_point == 'Y': continue
        
        for subject, preference in dict_preference[id].items():
            if system_A not in PerferenceValue:
                PerferenceValue[system_A] = []
            if system_B not in PerferenceValue:
                PerferenceValue[system_B] = []
                
            PerferenceValue[system_A].append(-preference)
            PerferenceValue[system_B].append(preference)
            count += 1
    
    keys = PerferenceValue.keys()
    print keys
    print stats_util.ttest(PerferenceValue[keys[0]], PerferenceValue[keys[1]], 1, 1)
    
    print count

def load_human_summary(cid, sheet, prompt, N):
    H = []
    for i in range(N):
        reffile = os.path.join('../../data/%s/ILP_MC/'%cid, str(sheet), prompt + '.ref.%d' %i)
        ref = load_summary(reffile)
        H.append(ref)
    return H

def load_system_sumary(cid, model, L, sheet, prompt, N):
    sumfile = None
    if model == 'MC': #need to get the lambda first
        lambda_json = "../../data/%s/MC/lambda_%d.json"%(cid, L)
        lambdas = fio.LoadDictJson(lambda_json)
        lambda_x = lambdas[str(sheet)][0]        
        sumfile = os.path.join('../../data/%s/%s'%(cid, 'ILP_MC'), str(sheet), '%s.sentence.L%d.%s.0.0.summary'%(prompt, L, lambda_x))
    elif model == 'ILP':
        sumfile = os.path.join('../../data/%s/%s'%(cid, 'ILP_MC'), str(sheet), '%s.sentence.L%d.0.0.0.0.summary'%(prompt, L))
    else:
        sumfile = os.path.join('../../data/%s/%s'%(cid, model), str(sheet), '%s.summary'%(prompt))
    return load_summary(sumfile)

def get_prompts(cid, prompt):
    if cid in ['Engineer', 'IE256', 'IE256_2016', 'CS0445']:
        return prompts[prompt]
    return ''
    
def task_generator(cid, modelpairs, output):
    head = ['id', 'prompt', 'summary_human', 'summary_A', 'summary_B']
    loghead = ['id', 'prompt', 'week', 'system_A', 'system_B', 'checking_point']
    
    from config import ConfigFile
    config = ConfigFile(config_file_name='config_%s.txt'%cid)
      
    matrix_dir = "../../data/%s/MC/"%cid
    
    sheets = global_params.lectures[cid]
    types = config.get_types()
    N = global_params.no_human[cid]
    L = global_params.getLL(cid)[0]
    
    random.seed(177)
    numpy.random.seed(177)
        
    for k, modelpair in enumerate(modelpairs):
        count = 0
        nocount = 0
        checking_count = 0
        id = 0
        
        body = []
        logbody = []
    
        summaryA, summaryB = modelpair
        prefix = os.path.join(output, '%s_%d_%s_%s'%(cid, k, summaryA, summaryB))
        
        for prompt in types:     
            for sheet in sheets:
                week = sheet
                
                #read TA's summmary
                Hs = load_human_summary(cid, sheet, prompt, N)
                
                rp = random.random()
                if rp>0.5:#switch
                    count += 1
                    summaryA, summaryB = summaryB, summaryA
                else:
                    nocount += 1
                
                #read system summary
                A = load_system_sumary(cid, summaryA, L, sheet, prompt, N)
            
                B = load_system_sumary(cid, summaryB, L, sheet, prompt, N)
                
                for H in Hs:
                    A = sort_by_cosin(A, H)
                    B = sort_by_cosin(B, H)
                    
                    logrow = [str(id), prompt, week, summaryA, summaryB, 'N']
                    row = [str(id), '"%s"'%prompts[prompt], list2paragraph(H), list2paragraph(A), list2paragraph(B)]
                    id += 1
                    
                    body.append(row)
                    logbody.append(logrow)
                
                rp = random.random()
                if rp>1-1./20.0:#add a checking point
                    logrow, row = generate_checking_point(cid, sheets, prompt, week, summaryA, summaryB, N, L)
                    body.append([str(id)] + row)
                    logbody.append([str(id)] + logrow)
                    checking_count += 1
                    id += 1
    
        fio.WriteCSV(prefix + '.csv', body, head, ',')
        
        fio.WriteMatrix(prefix + '.log', logbody, loghead)
    
        print count, nocount, checking_count
            
if __name__ == '__main__':
    modelpairs = [('MC', 'ILP'), 
                  ('MC', 'SumBasic'), 
                  ('SumBasic', 'ILP'),
                  #('MC', 'MEAD'),
                  #('MEAD', 'SumBasic'),
                  ]
    
    for cid in ['Engineer',
                'IE256',
                'IE256_2016',
                'CS0445',
                'review_camera', 
                'review_IMDB', 
                'review_prHistory',
                'DUC04',
            ]:
            
        output = '../../data/%s/mtask'%cid
        fio.NewPath(output)
        
        task_generator(cid, modelpairs, output)
    
#     result_analyze('../../data/Engineer/done/input_0_MC_ILP.log', 
#                    '../../data/Engineer/done/input_0_MC_ILP.out', 
#                    '../../data/Engineer/done/input_0_MC_ILP.results.txt')
#     
#     result_analyze('../../data/Engineer/done/input_1_MC_Mead.log', 
#                    '../../data/Engineer/done/input_1_MC_Mead.out', 
#                    '../../data/Engineer/done/input_1_MC_Mead.results.txt')
#     
#     #result_analyze('../../data/Engineer/done/input_2_MC_Oracle_ILP.log', 
#     #               '../../data/Engineer/done/input_2_MC_Oracle_ILP.out', 
#     #               '../../data/Engineer/done/input_2_MC_Oracle_ILP.results.txt')
#      
#     #result_analyze('../../data/Engineer/done/input_2_MC_Oracle.log', 
#     #               '../../data/Engineer/done/input_2_MC_Oracle.out', 
#     #               '../../data/Engineer/done/input_2_MC_Oracle.results.txt')
#     
#     result_analyze('../../data/Engineer/done/input_3_ILP_Mead.log', 
#                    '../../data/Engineer/done/input_3_ILP_Mead.out', 
#                    '../../data/Engineer/done/input_3_ILP_Mead.results.txt')
#         
#     get_agreement('../../data/Engineer/done/input_0_MC_ILP.log', 
#                    '../../data/Engineer/done/input_0_MC_ILP.out', 
#                    '../../data/Engineer/done/input_0_MC_ILP.results.txt')
#     
#     get_agreement('../../data/Engineer/done/input_1_MC_Mead.log', 
#                    '../../data/Engineer/done/input_1_MC_Mead.out', 
#                    '../../data/Engineer/done/input_1_MC_Mead.results.txt')
#         
#     get_agreement('../../data/Engineer/done/input_3_ILP_Mead.log', 
#                    '../../data/Engineer/done/input_3_ILP_Mead.out', 
#                    '../../data/Engineer/done/input_3_ILP_Mead.results.txt')