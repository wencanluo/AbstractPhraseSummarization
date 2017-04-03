import ILP_MC
import global_params
import os
import sys
import fio

def getSparseRatioExample(svddir, lambda_json, sheets, types, prefixA=".org.softA", eps=0.9):
    lambdas = fio.LoadDictJson(lambda_json)

    for sheet in sheets:
        week = sheet
        dir = os.path.join(svddir, str(week))
        
        lambda_x = lambdas[str(week)][0]
        
        for type in types:
            svdfile = os.path.join(svddir, str(week), '%s.%s%s'%(type, lambda_x, prefixA))
            keyfile = os.path.join(svddir, str(week), type + ".sentence.key")
            
            A = ILP_MC.LoadMC(svdfile)
            sentences = fio.ReadFile(keyfile)
            
            for bigram, row in A.items():     
                if bigram.find('ssss') != -1: continue
                for i, x in enumerate(row):
                    if x >= eps and x != 1.0:
                        if len(sentences[i].strip().split()) > 50: continue
                        print x, '\t', bigram, '@', sentences[i].strip(), '\t@', sheet, '\t', type

def findExamples(cid):
    from config import ConfigFile
    config = ConfigFile(config_file_name='config_%s.txt'%cid)
      
    matrix_dir = "../../data/%s/MC/"%cid
    ilpdir = "../../data/%s/ILP_MC/"%cid
    
    sheets = global_params.lectures[cid]
    types = config.get_types()
    
    output = os.path.join(matrix_dir, 'mc_examples.txt')
    
    SavedStdOut = sys.stdout
    sys.stdout = open(output, 'w')
    
    LL = global_params.getLL(cid)
    for L in LL:
        #load lambda json
        lambda_json = "../../data/%s/MC/lambda_%d.json"%(cid, L)
        getSparseRatioExample(matrix_dir, lambda_json, sheets, types, prefixA=".softA")
        
    sys.stdout = SavedStdOut
                        
if __name__ == '__main__':
    
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
            
            print cid
            findExamples(cid)
    
    
    