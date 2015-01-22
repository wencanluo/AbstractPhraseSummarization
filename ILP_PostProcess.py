import OracleExperiment
import fio
import numpy
import json

import postProcess
            
if __name__ == '__main__':
    datadir = "../../data/ILP/" 
                
    models = []
    for np in ['syntax', 'chunk']:
        for L in [10, 15, 20, 25, 30, 35, 40, 45, 50]:
            model = str(np) + '.L' + str(L)
            models.append(model)
    
    postProcess.CombineRouges(models, datadir)
            
    print "done"
    