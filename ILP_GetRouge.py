import OracleExperiment
import fio
import numpy
import json

tmpdir = "../../data/tmp/"
RougeHeader = ['R1-R', 'R1-P', 'R1-F', 'R2-R', 'R2-P', 'R2-F', 'RSU4-R', 'RSU4-P', 'RSU4-F',]
RougeHeaderSplit = ['R1-R', 'R1-P', 'R1-F', 'R2-R', 'R2-P', 'R2-F', 'RSU4-R', 'RSU4-P', 'RSU4-F','R1-R', 'R1-P', 'R1-F', 'R2-R', 'R2-P', 'R2-F', 'RSU4-R', 'RSU4-P', 'RSU4-F','R1-R', 'R1-P', 'R1-F', 'R2-R', 'R2-P', 'R2-F', 'RSU4-R', 'RSU4-P', 'RSU4-F',]
RougeNames = ['ROUGE-1','ROUGE-2', 'ROUGE-SUX']

def getRouge(datadir, np, L, outputdir, Lambda):
    #sheets = range(0,1)
    sheets = range(0,12)
    
    body = []
    
    for i, sheet in enumerate(sheets):
        week = i + 1
        
        Cache = {}
        cachefile = datadir + str(week) + '/' + 'cache.json'
        print cachefile
        if fio.IsExist(cachefile):
            with open(cachefile, 'r') as fin:
                Cache = json.load(fin)
                
        for type in ['POI', 'MP', 'LP']:
            row = []
            row.append(week)
        
            #read TA's summmary
            reffile = datadir + str(week) + '/' + type + '.ref.summary'
            lines = fio.ReadFile(reffile)
            ref = [line.strip() for line in lines]
            
            if Lambda == None:
                sumfile = datadir + str(week) + '/' + type + '.' + str(np) + '.L' + str(L) + '.summary'
            else:
                sumfile = datadir + str(week) + '/' + type + '.' + str(np) + '.L' + str(L) + "." + str(Lambda) + '.summary'
            
            lines = fio.ReadFile(sumfile)
            TmpSum = [line.strip() for line in lines]
            
            cacheKey = OracleExperiment.getKey(ref, TmpSum)
            if cacheKey in Cache:
                scores = Cache[cacheKey]
                print "Hit"
            else:
                print "Miss", cacheKey
                print sumfile
                scores = OracleExperiment.getRouge(ref, TmpSum)
                Cache[cacheKey] = scores
            
            row = row + scores
            
            body.append(row)
            
        with open(cachefile, 'w') as outfile:
            json.dump(Cache, outfile, indent=2)
            
    header = ['week'] + RougeHeader    
    row = []
    row.append("average")
    for i in range(1, len(header)):
        scores = [float(xx[i]) for xx in body]
        row.append(numpy.mean(scores))
    body.append(row)
    
    if Lambda == None:
        fio.WriteMatrix(outputdir + "rouge." + str(np) + '.L' + str(L) + ".txt", body, header)
    else:
        fio.WriteMatrix(outputdir + "rouge." + str(np) + '.L' + str(L) + "." + str(Lambda) + ".txt", body, header)

def getRougeSplit(datadir, np, L, outputdir, Lambda):
    #sheets = range(0,1)
    sheets = range(0,12)
    
    body = []
    
    for i, sheet in enumerate(sheets):
        week = i + 1
        
        Cache = {}
        cachefile = datadir + str(week) + '/' + 'cache.json'
        print cachefile
        if fio.IsExist(cachefile):
            with open(cachefile, 'r') as fin:
                Cache = json.load(fin)
               
        row = []
        row.append(week) 
        
        for type in ['POI', 'MP', 'LP']:
            #read TA's summmary
            reffile = datadir + str(week) + '/' + type + '.ref.summary'
            lines = fio.ReadFile(reffile)
            ref = [line.strip() for line in lines]
            
            if Lambda == None:
                sumfile = datadir + str(week) + '/' + type + '.' + str(np) + '.L' + str(L) + '.summary'
            else:
                sumfile = datadir + str(week) + '/' + type + '.' + str(np) + '.L' + str(L) + "." + str(Lambda) + '.summary'
            
            lines = fio.ReadFile(sumfile)
            TmpSum = [line.strip() for line in lines]
            
            cacheKey = OracleExperiment.getKey(ref, TmpSum)
            if cacheKey in Cache:
                scores = Cache[cacheKey]
                print "Hit"
            else:
                print "Miss", cacheKey
                print sumfile
                scores = OracleExperiment.getRouge(ref, TmpSum)
                Cache[cacheKey] = scores
            
            row = row + scores
            
        body.append(row)
            
        with open(cachefile, 'w') as outfile:
            json.dump(Cache, outfile, indent=2)
            
    header = ['week'] + RougeHeaderSplit    
    row = []
    row.append("average")
    for i in range(1, len(header)):
        scores = [float(xx[i]) for xx in body]
        row.append(numpy.mean(scores))
    body.append(row)
    
    if Lambda == None:
        fio.WriteMatrix(outputdir + "rouge." + str(np) + '.L' + str(L) + ".txt", body, header)
    else:
        fio.WriteMatrix(outputdir + "rouge." + str(np) + '.L' + str(L) + "." + str(Lambda) + ".txt", body, header)
            
                        
if __name__ == '__main__':
#     datadir = "../../data/ILP1/" 
#      
#     for L in [10, 15, 20, 25, 30, 35, 40, 45, 50]:
#         for np in ['syntax', 'chunk']:
#             getRouge(datadir, np, L, datadir, Lambda = None)
                 
#     datadir = "../../data/ILP2/" 
#     
#     for Lambda in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
#         for L in [10, 15, 20, 25, 30, 35, 40, 45, 50]:
#             for np in ['syntax']:
#                 getRouge(datadir, np, L, datadir, Lambda)
     
#     #without LP
#     datadir = "../../data/ILP_Sentence_SVD/" 
#          
#     for L in [30]:
#         for np in ['sentence']:
#             getRougeSplit(datadir, np, L, datadir, Lambda = None)
#   
#     ilpdir = "../../data/ILP2/"
#       
#     #for Lambda in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
#     for Lambda in [0.8]:
#         #for L in [10, 15, 20, 25, 30, 35, 40, 45, 50]:
#         for L in [30]:
#             for np in ['syntax', ]: #'chunk'
#                 getRouge(ilpdir, np, L, ilpdir, Lambda)
#                     
#     ilpdir = "../../data/ILP_Unibigram/"
#       
#     #for Lambda in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
#     for Lambda in [0.8]:
#         #for L in [10, 15, 20, 25, 30, 35, 40, 45, 50]:
#         for L in [30]:
#             for np in ['syntax', ]: #'chunk'
#                 getRouge(ilpdir, np, L, ilpdir, Lambda)
#       
#                      
#     ilpdir = "../../data/ILP_UnibigramMalformedFilter/"
#      
#     #for Lambda in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
#     for Lambda in [0.8]:
#         #for L in [10, 15, 20, 25, 30, 35, 40, 45, 50]:
#         for L in [30]:
#             for np in ['syntax', ]: #'chunk'
#                 getRouge(ilpdir, np, L, ilpdir, Lambda)
#      
#     
#     ilpdir = "../../data/ILP2_MalformedFilter/" 
#     #for Lambda in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
#     for Lambda in [0.8]:
#         #for L in [10, 15, 20, 25, 30, 35, 40, 45, 50]:
#         for L in [30]:
#             for np in ['syntax', ]: #'chunk'
#                 getRouge(ilpdir, np, L, ilpdir, Lambda)
    
    ilpdir = "../../data/ILP_Sentence_Supervised_FeatureWeightingMC/" 
    #ilpdir = "../../data/ILP_Sentence_Supervised/" 
    
    for Lambda in [1.0]:
        #for L in [10, 15, 20, 25, 30, 35, 40, 45, 50]:
        for L in [30]:
            for np in ['sentence', ]: #'chunk'
                getRougeSplit(ilpdir, np, L, ilpdir, Lambda)
    
    #datadir = "../../data/ILP1_Sentence/" 
         
#     for L in [30]:
#         for np in ['sentence']:
#             getRougeSplit(datadir, np, L, datadir, Lambda = None)
                          
    print "done"
    