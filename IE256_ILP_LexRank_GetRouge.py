import OracleExperiment
import fio
import numpy
import json
import os
from TAC_ILP_baseline import iter_folder
from numpy import average
import codecs
import global_params
import Survey
from  OracleExperiment import RougeHeader as RougeHeader
from  OracleExperiment import RougeNames as RougeNames

tmpdir = "../../data/tmp/"
RougeHeaderSplit = ['R1-R', 'R1-P', 'R1-F', 'R2-R', 'R2-P', 'R2-F', 'RSU4-R', 'RSU4-P', 'RSU4-F','R1-R', 'R1-P', 'R1-F', 'R2-R', 'R2-P', 'R2-F', 'RSU4-R', 'RSU4-P', 'RSU4-F','R1-R', 'R1-P', 'R1-F', 'R2-R', 'R2-P', 'R2-F', 'RSU4-R', 'RSU4-P', 'RSU4-F',]

def getRouge(rouge_dict, ilpdir, L, outputdir, Lambda, sheets, types, N=2):
    body = []
    
    for sheet in sheets:
        week = sheet
        dir = ilpdir + str(week) + '/'
        
        for type in types:
            prefix = dir + type

            summary_file = prefix + ".summary"
                            
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
            for i in range(N):
                reffile = os.path.join(ilpdir, str(week), type + '.ref.%d' %i)
                if not fio.IsExist(reffile):
                    print reffile
                    continue
                    
                lines = fio.ReadFile(reffile)
                ref = [line.strip() for line in lines]
                refs.append(ref)
            
            if len(refs) == 0: continue
              
            lstref = []
            for ref in refs:
                lstref += ref
            
            lines = fio.ReadFile(summary_file)
            TmpSum = [Survey.NormalizeMeadSummary(line) for line in lines]
        
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
            except Exception as e:
                print e
                pass
        
    header = ['id'] + RougeHeader
    row = ['ave']
    for i in range(1, len(header)):
        scores = [float(xx[i]) for xx in body]
        row.append(numpy.mean(scores))
    body.append(row)
    
    fio.WriteMatrix(os.path.join(outputdir, "rouge.sentence." + 'L' + str(L) + ".txt"), body, header)


def getBaselineROUGE(cid):
    ilpdir = "../../data/%s/ILP_Baseline/"%cid
    sheets = global_params.lectures[cid]
    
    rouge_dict = {}
    
    for L in [10, 15, 20, 25, 30, 35, 40]:
        Lambda = None
                
        getRouge(rouge_dict, ilpdir, L, ilpdir, Lambda, sheets)
                
        fio.SaveDict2Json(rouge_dict, ilpdir + 'rouge.L'+str(L)+'.json')
            
def gatherRouge():
    
    Allbody = []
    for cid in [
                'Engineer',
                'IE256',
                'IE256_2016',
                'CS0445',
                'review_camera',
                'review_IMDB', 
                'review_prHistory',
                'DUC04',
                ]:
                
        LL = global_params.getLL(cid)
            
        for L in LL:
            ilpdir = "../../data/%s/"%cid
            rougefile = os.path.join(ilpdir, 'test_%d.txt'%L)
            if not fio.IsExist(rougefile): continue
            ilphead, ilpbody = fio.ReadMatrix(rougefile, hasHead=True)
            
            for model in ['MEAD', 
                          'LexRank', 
                          'SumBasic']:
                
                baseline_dir = "../../data/%s/%s/"%(cid,model)
                baseline_rougefile = os.path.join(baseline_dir, 'rouge.sentence.L%d.txt'%L)
                
                if not fio.IsExist(baseline_rougefile): continue
                
                head, body = fio.ReadMatrix(baseline_rougefile, hasHead=True)
                
                cidname = global_params.mapcid(cid)
                row = [cidname, model] + ['%.3f'%float(x) for x in body[-1][1:-3]]
                
                print cid, L, model
                print rougefile
                print baseline_rougefile
                #get p values
                from exp_select_lambda import get_ttest_pvalues
                pvalues = get_ttest_pvalues(ilpbody[1:-2], body[1:-1], range(1,len(head)-3))
                print pvalues
                
                k = 2
                for p in pvalues:
                    if p < 0.05:
                        row[k] = row[k]+'$^*$'
                    k+=1
                
                Allbody.append(row)

    output = '../../rouge_all_gather_baselines.txt'
    fio.Write2Latex(output, Allbody, ['Corpus'] + head)
                            
if __name__ == '__main__':
    import sys
    
#     gatherRouge()
#     exit(0)

    for cid in [
                'Engineer', 
                'IE256',
                'IE256_2016',
                'CS0445',
                'review_camera', 
                'review_IMDB', 
                'review_prHistory',
                'DUC04',
                 ]:
        
        sheets = global_params.lectures[cid]
        N = global_params.no_human[cid]
        from config import ConfigFile
        config = ConfigFile(config_file_name='config_%s.txt'%cid)
        types = config.get_types()
        LL = global_params.getLL(cid)
        
        for model in ['MEAD', 'LexRank', 'SumBasic']:
            ilpdir = "../../data/%s/%s/"%(cid,model)
            
            rouge_dict = {}
                
            for L in LL:
                print ilpdir, L 
                Lambda = None
                
                getRouge(rouge_dict, ilpdir, L, ilpdir, Lambda, sheets, types, N)
                fio.SaveDict2Json(rouge_dict, ilpdir + 'rouge.L'+str(L)+'.json')
               
    print "done"
