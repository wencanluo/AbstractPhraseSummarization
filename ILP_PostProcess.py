import OracleExperiment
import fio
import numpy
import json

import postProcess
            
if __name__ == '__main__':
#     datadir = "../../data/ILP/" 
#                 
#     models = []
#     for np in ['syntax', 'chunk']:
#         for L in [10, 15, 20, 25, 30, 35, 40, 45, 50]:
#             model = str(np) + '.L' + str(L)
#             models.append(model)
#     
#     postProcess.CombineRouges(models, datadir)

    datadir = "../../data/ILP2/" 
                 
    models = []
    
    for np in ['syntax']:
        for Lambda in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
            for L in [10, 15, 20, 25, 30, 35, 40, 45, 50]:
                model = str(np) + '.L' + str(L) + "." + str(Lambda)
                models.append(model)
     
    postProcess.CombineRouges(models, datadir)
            
    print "done"
    