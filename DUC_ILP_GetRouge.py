import OracleExperiment
import fio
import numpy
import json
import os
from TAC_ILP_baseline import iter_folder
from numpy import average
import codecs

tmpdir = "../../data/tmp/"
RougeHeader = ['R1-R', 'R1-P', 'R1-F', 'R2-R', 'R2-P', 'R2-F', 'RSU4-R', 'RSU4-P', 'RSU4-F',]
RougeHeaderSplit = ['R1-R', 'R1-P', 'R1-F', 'R2-R', 'R2-P', 'R2-F', 'RSU4-R', 'RSU4-P', 'RSU4-F','R1-R', 'R1-P', 'R1-F', 'R2-R', 'R2-P', 'R2-F', 'RSU4-R', 'RSU4-P', 'RSU4-F','R1-R', 'R1-P', 'R1-F', 'R2-R', 'R2-P', 'R2-F', 'RSU4-R', 'RSU4-P', 'RSU4-F',]
RougeNames = ['ROUGE-1','ROUGE-2', 'ROUGE-SUX']

def getRouge(rouge_dict, datadir, years, L, outputdir, Lambda):
    
    average_body = []
    
    bodys = []
    
    for year in years:
        body = []
        
        path = os.path.join(ilpdir, year)
        print path
        
        Cache = {}
        cachefile = os.path.join(path, 'cache.json')
        
        print cachefile
        if fio.IsExist(cachefile):
            try:
                with open(cachefile, 'r') as fin:
                    Cache = json.load(fin)
            except:
                Cache = {}
                
        for subdir, file in iter_folder(path, '.key'):
            doc_id = file[:-4]
            if doc_id == 'new': continue
            
            row = [doc_id]
            prefix = os.path.join(path, doc_id)
            print prefix
            
            if Lambda == None:
                sumfile = prefix + '.L' + str(L) + '.summary'
            else:
                sumfile = prefix + '.L' + str(L) + "." + str(Lambda) + '.summary'
            
            if not fio.IsExist(sumfile): 
                print sumfile
                continue
            
            #read TA's summmary
            refs = []
            for i in range(4):
                reffile = prefix + '.ref%d.summary' %(i+1)
                lines = fio.ReadFile(reffile)
                ref = [line.strip() for line in lines]
                refs.append(ref)
            lstref = refs[0] + refs[1] + refs[2] + refs[3]  
            
            lines = fio.ReadFile(sumfile)
            TmpSum = [line.strip() for line in lines]
            
            cacheKey = OracleExperiment.getKey(lstref, TmpSum)
            if cacheKey in Cache:
                scores = Cache[cacheKey]
                print "Hit"
            else:
                print "Miss"
                print sumfile
                scores = OracleExperiment.getRouge_Tac(refs, TmpSum)
                Cache[cacheKey] = scores
            
            rouge_dict[doc_id] = scores
            
            row = row + scores
            
            body.append(row)
            #break
        
        bodys += body
        
        try:
            fio.SaveDict2Json(Cache, cachefile)
        except:
            #fio.SaveDict(Cache, cachefile + '.dict')
            pass
        
        header = ['id'] + RougeHeader
        row = []
        row.append(year)
        for i in range(1, len(header)):
            scores = [float(xx[i]) for xx in body]
            row.append(numpy.mean(scores))
        average_body.append(row)
    
    if Lambda == None:
        fio.WriteMatrix(os.path.join(outputdir, "rouge." + 'L' + str(L) + ".txt"), bodys, header)
    else:
        fio.WriteMatrix(os.path.join(outputdir, "rouge." + 'L' + str(L) + "." + str(Lambda) + ".txt"), bodys, header)
    
    if Lambda == None:
        fio.WriteMatrix(os.path.join(outputdir, "average_rouge." + 'L' + str(L) + ".txt"), average_body, header)
    else:
        fio.WriteMatrix(os.path.join(outputdir, "average_rouge." + 'L' + str(L) + "." + str(Lambda) + ".txt"), average_body, header)
    
    
if __name__ == '__main__':
    import sys
    
    ilpdir = sys.argv[1]
    m_lambda = sys.argv[2]
    threshold = sys.argv[3]
    
    #ilpdir = '../../data/TAC_ILP_Sentence/'
    print ilpdir
    
    from config import ConfigFile
    config = ConfigFile(config_file_name='tac_config.txt')
                    
    years = ['DUC_2004', 
             ]
    
    rouge_dict = {}
    
    for L in [config.get_length_limit()]:
        #for threshold in [0.0]:
        #for m_lambda in ['2']: 
        Lambda =  str(m_lambda)+ '.' + str(threshold)
        getRouge(rouge_dict, ilpdir, years, L, ilpdir, Lambda)
    
        fio.SaveDict2Json(rouge_dict, ilpdir + 'rouge.L'+str(L)+'.'+Lambda+'.json')
                     
    print "done"
