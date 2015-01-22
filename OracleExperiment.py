import fio
import copy
import subprocess
import json
import numpy

tmpdir = "../../data/tmp/"
RougeHeader = ['R1-R', 'R1-P', 'R1-F', 'R2-R', 'R2-P', 'R2-F', 'RSU4-R', 'RSU4-P', 'RSU4-F',]
RougeNames = ['ROUGE-1','ROUGE-2', 'ROUGE-SUX']

def getRouge(ref, model):
    #return the Rouge scores given the reference summary and the models
    
    #write the files
    fio.SaveList(ref, tmpdir+'ref.txt', '\n')
    fio.SaveList(model, tmpdir+'model.txt', '\n')
    
    retcode = subprocess.call(['./get_rouge'], shell=True)
    if retcode != 0:
        print("Failed!")
        exit(-1)
    else:
        print "Passed!"
    
    row = []
    for scorename in RougeNames:
        filename = tmpdir + "OUT_"+scorename+".csv"
        lines = fio.readfile(filename)
        try:
            scorevalues = lines[1].split(',')
            score = scorevalues[1].strip()
            row.append(score)
            score = scorevalues[2].strip()
            row.append(score)
            score = scorevalues[3].strip()
            row.append(score)
        except Exception:
            print filename, scorename, lines
            
    return row

def getKey(ref, model):
    return "@".join(ref) + ":" + "@".join(model)
    
def Greedy(oracledir, np, L, metric='R1-F'):
    #sheets = range(0,1)
    sheets = range(0,12)
    RIndex = RougeHeader.index(metric)
    assert(RIndex != -1)
    
    for i, sheet in enumerate(sheets):
        week = i + 1
        
        #Add a cache to make it faster
        Cache = {}
        cachefile = oracledir + str(week) + '/' + 'cache.json'
        if fio.isExist(cachefile):
            with open(cachefile, 'r') as fin:
                Cache = json.load(fin)
        
        #for type in ['POI']:
        for type in ['POI', 'MP', 'LP']:
            #read TA's summmary
            reffile = oracledir + str(week) + '/' + type + '.ref.summary'
            lines = fio.readfile(reffile)
            ref = [line.strip() for line in lines]
            
            #read Phrases
            phrasefile = oracledir + str(week) + '/' + type + '.' + str(np) + '.key'
            lines = fio.readfile(phrasefile)
            candidates = [line.strip() for line in lines]
            
            summary = []
            Length = 0
            
            maxSum = []
            maxScore = 0
            Round = 1
            
            Changed = True
            while Changed:
                Changed = False
                for phrase in candidates:
                    WC = len(phrase.split())
                    if Length + WC > L: continue
                    
                    TmpSum = copy.deepcopy(summary)
                    TmpSum.append(phrase)
                    
                    #get Rouge Score
                    cacheKey = getKey(ref, TmpSum)
                    if cacheKey in Cache:
                        scores = Cache[cacheKey]
                        print "Hit"
                    else:
                        scores = getRouge(ref, TmpSum)
                        Cache[cacheKey] = scores
                    
                    s = float(scores[RIndex])
                    #s = scores[RIndex]
                    if s > maxScore:
                        maxSum = TmpSum
                        maxScore = s
                        Changed = True
                
                if Changed:
                    #write the results
                    sumfile = oracledir + str(week) + '/' + type + '.' + str(np) + '.L' + str(L) + "." + str(metric) + '.R' + str(Round) +'.summary'
                    fio.SaveList(maxSum, sumfile, '\r\n')
                    
                    summary = maxSum
                    Length = 0
                    for s in maxSum:
                        Length = Length + len(s.split())
                    
                    Round = Round + 1
                    
                    newCandidates = []
                    #remove the candidate from the existing summary
                    for phrase in candidates:
                        if phrase not in maxSum:
                            newCandidates.append(phrase)
                    
                    candidates = newCandidates

        with open(cachefile, 'w') as outfile:
            json.dump(Cache, outfile, indent=2)

def getOracleRouge(oracledir, np, L, metric, outputdir):
    #sheets = range(0,1)
    sheets = range(0,12)
    
    body = []
    
    for i, sheet in enumerate(sheets):
        week = i + 1
            
        #Add a cache to make it faster
        Cache = {}
        cachefile = oracledir + str(week) + '/' + 'cache.json'
        print cachefile
        if fio.isExist(cachefile):
            with open(cachefile, 'r') as fin:
                Cache = json.load(fin)
        
        for type in ['POI', 'MP', 'LP']:
            row = []
            row.append(week)
        
            #read TA's summmary
            reffile = oracledir + str(week) + '/' + type + '.ref.summary'
            lines = fio.readfile(reffile)
            ref = [line.strip() for line in lines]
            
            Round = 1
            while True:
                sumfile = oracledir + str(week) + '/' + type + '.' + str(np) + '.L' + str(L) + "." + str(metric) + '.R' + str(Round) +'.summary'
                if not fio.isExist(sumfile): break
                Round = Round + 1
            
            Round = Round - 1
            sumfile = oracledir + str(week) + '/' + type + '.' + str(np) + '.L' + str(L) + "." + str(metric) + '.R' + str(Round) +'.summary'
            
            lines = fio.readfile(sumfile)
            TmpSum = [line.strip() for line in lines]
            
            cacheKey = getKey(ref, TmpSum)
            if cacheKey in Cache:
                scores = Cache[cacheKey]
                print "Hit"
            else:
                print "Miss", cacheKey
                print sumfile
                scores = getRouge(ref, TmpSum)
                Cache[cacheKey] = scores
                #exit()
            
            row = row + scores
            
            body.append(row)
            
    header = ['week'] + RougeHeader    
    row = []
    row.append("average")
    for i in range(1, len(header)):
        scores = [float(xx[i]) for xx in body]
        row.append(numpy.mean(scores))
    body.append(row)
    
    fio.writeMatrix(outputdir + "rouge." + str(np) + '.L' + str(L) + "." + str(metric) + ".txt", body, header)
             
def TestRouge():
    ref = ["police killed the gunman"]
    S1 = ["police kill the gunman"]
    S2 = ["the gunman kill police"]
    
    getRouge(ref, S2)
    
if __name__ == '__main__':
    oracledir = "../../data/oracle/" 
    datadir = "../../data/oracle/"
    #TestRouge()
    
    for L in [10, 15, 20, 25, 30, 35, 40, 45, 50]:
        for np in ['syntax', 'chunk']:
            for metric in ['R1-F', 'R2-F', 'RSU4-F']:
                Greedy(oracledir, np, L, metric)
    
    for L in [10, 15, 20, 25, 30, 35, 40, 45, 50]:
        for np in ['syntax', 'chunk']:
            for metric in ['R1-F', 'R2-F', 'RSU4-F']:
                getOracleRouge(oracledir, np, L, metric, datadir)
    
            
    print "done"
    