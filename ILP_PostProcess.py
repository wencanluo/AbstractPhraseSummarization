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

    datadir = "../../data/TAC_ILP_Sentence_MC/" 
                 
    models = []
    
    for np in ['sentence']:
        for threshold in [0.2, 0.4, 0.6, 0.8, 1.0, 0.0]:
            for Lambda in ['2', '5']:
                for L in [100]:
                    model = 'L' + str(L) + "." + str(Lambda) + '.' + str(threshold)
                    models.append(model)
     
    postProcess.CombineRouges(models, datadir, prefix='average_rouge')
            
    print "done"
    