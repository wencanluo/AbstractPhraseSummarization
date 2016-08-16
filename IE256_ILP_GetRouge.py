import OracleExperiment
import fio
import numpy
import json
import os
from TAC_ILP_baseline import iter_folder
from numpy import average
import codecs
import global_params
from _ast import Lambda

tmpdir = "../../data/tmp/"
RougeHeader = ['R1-R', 'R1-P', 'R1-F', 'R2-R', 'R2-P', 'R2-F', 'RSU4-R', 'RSU4-P', 'RSU4-F',]
RougeHeaderSplit = ['R1-R', 'R1-P', 'R1-F', 'R2-R', 'R2-P', 'R2-F', 'RSU4-R', 'RSU4-P', 'RSU4-F','R1-R', 'R1-P', 'R1-F', 'R2-R', 'R2-P', 'R2-F', 'RSU4-R', 'RSU4-P', 'RSU4-F','R1-R', 'R1-P', 'R1-F', 'R2-R', 'R2-P', 'R2-F', 'RSU4-R', 'RSU4-P', 'RSU4-F',]
RougeNames = ['ROUGE-1','ROUGE-2', 'ROUGE-SUX']

def getRouge(rouge_dict, ilpdir, L, outputdir, Lambda, sheets):
    body = []
    
    for sheet in sheets:
        week = sheet
        dir = ilpdir + str(week) + '/'
        
        for type in ['q1', 'q2']:
            prefix = dir + type + "." + 'sentence'
            
            if Lambda == None:
                summary_file = prefix + '.L' + str(L) + ".summary"
            else:
                summary_file = prefix + '.L' + str(L) + '.'+str(Lambda) + ".summary"
                
            print summary_file
            
            if not fio.IsExist(summary_file): 
                print summary_file
                continue
            
            Cache = {}
            cachefile = os.path.join(ilpdir, str(week), 'cache.json')
            print cachefile
            if fio.IsExist(cachefile):
                with open(cachefile, 'r') as fin:
                    Cache = json.load(fin)
            
            #read TA's summmary
            refs = []
            for i in range(2):
                reffile = os.path.join(ilpdir, str(week), type + '.ref.%d' %i)
                if not fio.IsExist(reffile):
                    print reffile
                    continue
                    
                lines = fio.ReadFile(reffile)
                ref = [line.strip() for line in lines]
                refs.append(ref)
            
            if len(refs) == 0: continue
              
            lstref = refs[0] + refs[1]
        
            lines = fio.ReadFile(summary_file)
            TmpSum = [line.strip() for line in lines]
        
            cacheKey = OracleExperiment.getKey(lstref, TmpSum)
            if cacheKey in Cache:
                scores = Cache[cacheKey]
                print "Hit"
            else:
                print "Miss"
                print summary_file
                scores = OracleExperiment.getRouge_IE256(refs, TmpSum)
                Cache[cacheKey] = scores
            
            rouge_dict[str(week)+'_'+type] = scores
            row = [week]
            row = row + scores
            
            body.append(row)
        
            try:
                fio.SaveDict2Json(Cache, cachefile)
            except:
                #fio.SaveDict(Cache, cachefile + '.dict')
                pass
        
    header = ['id'] + RougeHeader
    row = ['ave']
    for i in range(1, len(header)):
        scores = [float(xx[i]) for xx in body]
        row.append(numpy.mean(scores))
    body.append(row)
    
    fio.WriteMatrix(os.path.join(outputdir, "rouge.sentence." + 'L' + str(L) + '.' + str(Lambda) + ".txt"), body, header)


def getBaselineROUGE(cid):
    ilpdir = "../../data/%s/ILP_Baseline/"%cid
    sheets = global_params.lectures[cid]
    
    rouge_dict = {}
    
    for L in [10, 15, 20, 25, 30, 35, 40]:
        Lambda = None
                
        getRouge(rouge_dict, ilpdir, L, ilpdir, Lambda, sheets)
                
        fio.SaveDict2Json(rouge_dict, ilpdir + 'rouge.L'+str(L)+'.json')
                        
if __name__ == '__main__':
    import sys
    
#     getBaselineROUGE('IE256')
#     exit(-1)
#     
    #ilpdir = sys.argv[1]
    
    #cid = 'CS0445'
    for cid in ['IE256']: #'IE256_2016', 'CS0445', 
        
        ilpdir = "../../data/%s/ILP_MC/"%cid
        sheets = global_params.lectures[cid]
        
        #m_lambda = 'org'
        #threshold = '1.0'
        
        from config import ConfigFile
        config = ConfigFile(config_file_name='config_%s.txt'%cid)
                        
        rouge_dict = {}
        
        for L in [10, 15, 20, 25, 30, 35, 40]:
        #for L in [39]:
            for threshold in [0.0]:
                for m_lambda in numpy.arange(0, 8.0, 0.1):
                #for m_lambda in numpy.arange(0.5, 6.0, 0.5):
                #for m_lambda in [0.0]:
            
                    if m_lambda == 'None':
                        Lambda = None
                    else:
                        Lambda =  str(m_lambda)+ '.' + str(threshold)
                    
                    getRouge(rouge_dict, ilpdir, L, ilpdir, Lambda, sheets)
                    
                    if m_lambda == 'None':
                        fio.SaveDict2Json(rouge_dict, ilpdir + 'rouge.L'+str(L)+'.json')
                    else:
                        fio.SaveDict2Json(rouge_dict, ilpdir + 'rouge.L'+str(L)+'.'+Lambda+'.json')
                         
    print "done"
