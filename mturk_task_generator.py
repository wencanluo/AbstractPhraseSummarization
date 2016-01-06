import fio
import numpy
import json
import random
import re
import codecs
import Cosin
from collections import defaultdict
import stats_util

folders = {'Mead':'../../data/Engineer/Mead/',
           'Oracle_ILP':'../../data/Engineer/ILP_Oracle/',
           'Oracle':'../../data/Engineer/Oracle/',
           'ILP':'../../data/Engineer/ILP_Baseline/',
           'MC':'../../data/Engineer/ILP_MC/',
           }

prompts = {'MP':"Describe what was confusing or needed more detail.",
           'POI':"Describe what you found most interesting in today's class.",
           'LP':"Describe what you learned about how you learn.",
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

def generate_checking_point(prompt, week, summaryA, summaryB):
    
    w = random.randint(1, 12)
    while w == week:
        w = random.randint(1, 12)
    
    rp = random.random()
    if rp>0.5:#switch
        summaryB = summaryA
    else:
        summaryA = summaryB
    
    #read TA's summmary
    reffile = folders['ILP'] + str(w) + '/' + prompt + '.ref.summary'
    H = load_summary(reffile)
                
    sumfileA = folders[summaryA] + str(w) + '/' + prompt + '.' + str('sentence') + '.L' + str(30) + '.summary'
    A = load_summary(sumfileA)

    sumfileB = folders[summaryB] + str(w) + '/' + prompt + '.' + str('sentence') + '.L' + str(30) + '.summary'
    B = load_summary(sumfileB)
    
    p = numpy.random.permutation(len(B))
    
    B = [B[i] for i in p]
    
    logrow = [prompt, w, summaryA, summaryB, 'Y']
    row = ['"%s"'%prompts[prompt], list2paragraph(H), list2paragraph(A), list2paragraph(B)]
                    
    return logrow, row
    
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
    
def task_generator(modelpairs, output):
    head = ['id', 'prompt', 'summary_human', 'summary_A', 'summary_B']
    loghead = ['id', 'prompt', 'week', 'system_A', 'system_B', 'checking_point']
    
    sheets = range(0,12)
    for k, modelpair in enumerate(modelpairs):
        random.seed(1)
        numpy.random.seed(0)
    
        count = 0
        nocount = 0
        checking_count = 0
        id = 0
        
        body = []
        logbody = []
    
        summaryA, summaryB = modelpair
        prefix = output + str(k) + '_' + summaryA + '_' + summaryB
        
        for type in ['POI', 'MP', 'LP']:     
            for i, sheet in enumerate(sheets):
                week = i + 1
                
                #read TA's summmary
                reffile = folders['ILP'] + str(week) + '/' + type + '.ref.summary'
                H = load_summary(reffile)
                
                rp = random.random()
                if rp>0.5:#switch
                    count += 1
                    summaryA, summaryB = summaryB, summaryA
                else:
                    nocount += 1
                    
                sumfileA = folders[summaryA] + str(week) + '/' + type + '.' + str('sentence') + '.L' + str(30) + '.summary'
                A = load_summary(sumfileA)
            
                sumfileB = folders[summaryB] + str(week) + '/' + type + '.' + str('sentence') + '.L' + str(30) + '.summary'
                B = load_summary(sumfileB)
                
                A = sort_by_cosin(A, H)
                B = sort_by_cosin(B, H)
                
                logrow = [str(id), type, week, summaryA, summaryB, 'N']
                row = [str(id), '"%s"'%prompts[type], list2paragraph(H), list2paragraph(A), list2paragraph(B)]
                id += 1
                
                body.append(row)
                logbody.append(logrow)
                
                rp = random.random()
                if rp>1-1./7.0:#add a checking point
                    logrow, row = generate_checking_point(type, week, summaryA, summaryB)
                    body.append([str(id)] + row)
                    logbody.append([str(id)] + logrow)
                    checking_count += 1
                    id += 1
    
        fio.WriteCSV(prefix + '.csv', body, head, ',')
        
        fio.WriteMatrix(prefix + '.log', logbody, loghead)
    
        print count, nocount, checking_count
            
if __name__ == '__main__':
    modelpairs = [('MC', 'ILP'), 
                  ('MC', 'Mead'), 
                  ('MC', 'Oracle'),
                  ('ILP', 'Mead'), 
                  ('ILP', 'Oracle'),
                  ('Mead', 'Oracle'),
                  ]
    
    output = '../../data/Engineer/input_'
    #task_generator(modelpairs, output)
    
    result_analyze('../../data/Engineer/done/input_0_MC_ILP.log', 
                   '../../data/Engineer/done/input_0_MC_ILP.out', 
                   '../../data/Engineer/done/input_0_MC_ILP.results.txt')
    
    result_analyze('../../data/Engineer/done/input_1_MC_Mead.log', 
                   '../../data/Engineer/done/input_1_MC_Mead.out', 
                   '../../data/Engineer/done/input_1_MC_Mead.results.txt')
    
    result_analyze('../../data/Engineer/done/input_2_MC_Oracle_ILP.log', 
                   '../../data/Engineer/done/input_2_MC_Oracle_ILP.out', 
                   '../../data/Engineer/done/input_2_MC_Oracle_ILP.results.txt')
     
    result_analyze('../../data/Engineer/done/input_2_MC_Oracle.log', 
                   '../../data/Engineer/done/input_2_MC_Oracle.out', 
                   '../../data/Engineer/done/input_2_MC_Oracle.results.txt')
    
    result_analyze('../../data/Engineer/done/input_3_ILP_Mead.log', 
                   '../../data/Engineer/done/input_3_ILP_Mead.out', 
                   '../../data/Engineer/done/input_3_ILP_Mead.results.txt')