import fio
import copy
import subprocess
import json
import numpy
import cmd

tmpdir = "../../data/tmp/"
RougeHeader = ['R1-R', 'R1-P', 'R1-F', 'R2-R', 'R2-P', 'R2-F', 'RSU4-R', 'RSU4-P', 'RSU4-F',]
RougeNames = ['ROUGE-1','ROUGE-2', 'ROUGE-SUX']

def getRouge_IE256(refs, model):
    #return the Rouge scores given the reference summary and the models
    
    #write the files
    fio.SaveList(model, tmpdir+'model.txt', '\n')
    
    for i, ref in enumerate(refs):
        fio.SaveList(ref, tmpdir+'ref%d.txt'%(i+1), '\n')
    
    N = len(refs)
    retcode = subprocess.call(['./get_rouge_ie256_%d'%N], shell=True)
    if retcode != 0:
        print("Failed!")
        exit(-1)
    else:
        print "Passed!"
    
    row = []
    for scorename in RougeNames:
        filename = tmpdir + "OUT_"+scorename+".csv"
        lines = fio.ReadFile(filename)
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

def getRouge_Tac(refs, model):
    #return the Rouge scores given the reference summary and the models
    
    #write the files
    fio.SaveList(model, tmpdir+'model.txt', '\n')
    
    for i, ref in enumerate(refs):
        fio.SaveList(ref, tmpdir+'ref%d.txt'%(i+1), '\n')
    
    retcode = subprocess.call(['./get_rouge_tac'], shell=True)
    if retcode != 0:
        print("Failed!")
        exit(-1)
    else:
        print "Passed!"
    
    row = []
    for scorename in RougeNames:
        filename = tmpdir + "OUT_"+scorename+".csv"
        lines = fio.ReadFile(filename)
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
        lines = fio.ReadFile(filename)
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
        if fio.IsExist(cachefile):
            with open(cachefile, 'r') as fin:
                Cache = json.load(fin)
        
        #for type in ['POI']:
        for type in ['POI', 'MP', 'LP']:
            #read TA's summmary
            reffile = oracledir + str(week) + '/' + type + '.ref.summary'
            lines = fio.ReadFile(reffile)
            ref = [line.strip() for line in lines]
            
            #read Phrases
            phrasefile = oracledir + str(week) + '/' + type + '.' + str(np) + '.key'
            lines = fio.ReadFile(phrasefile)
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
        if fio.IsExist(cachefile):
            with open(cachefile, 'r') as fin:
                Cache = json.load(fin)
        
        for type in ['POI', 'MP', 'LP']:
            row = []
            row.append(week)
        
            #read TA's summmary
            reffile = oracledir + str(week) + '/' + type + '.ref.summary'
            lines = fio.ReadFile(reffile)
            ref = [line.strip() for line in lines]
            
            Round = 1
            while True:
                sumfile = oracledir + str(week) + '/' + type + '.' + str(np) + '.L' + str(L) + "." + str(metric) + '.R' + str(Round) +'.summary'
                if not fio.IsExist(sumfile): break
                Round = Round + 1
            
            Round = Round - 1
            sumfile = oracledir + str(week) + '/' + type + '.' + str(np) + '.L' + str(L) + "." + str(metric) + '.R' + str(Round) +'.summary'
            
            if not fio.IsExist(sumfile):
                lines = []
            else:
                lines = fio.ReadFile(sumfile)
            TmpSum = [line.strip() for line in lines]
            
            newsumfile = oracledir + str(week) + '/' + type + '.' + str(np) + '.L' + str(L) +'.summary'
            fio.SaveList(TmpSum, newsumfile)
            
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
    
    fio.WriteMatrix(outputdir + "rouge." + str(np) + '.L' + str(L) + "." + str(metric) + ".txt", body, header)

def getOracleRougeSplit(oracledir, np, L, metric, outputdir):
    #sheets = range(0,1)
    sheets = range(0,12)
    
    body = []
    
    for i, sheet in enumerate(sheets):
        week = i + 1
            
        #Add a cache to make it faster
        Cache = {}
        cachefile = oracledir + str(week) + '/' + 'cache.json'
        print cachefile
        if fio.IsExist(cachefile):
            with open(cachefile, 'r') as fin:
                Cache = json.load(fin)
        
        row = []
        for type in ['POI', 'MP', 'LP']:
            row.append(week)
        
            #read TA's summmary
            reffile = oracledir + str(week) + '/' + type + '.ref.summary'
            lines = fio.ReadFile(reffile)
            ref = [line.strip() for line in lines]
            
            Round = 1
            while True:
                sumfile = oracledir + str(week) + '/' + type + '.' + str(np) + '.L' + str(L) + "." + str(metric) + '.R' + str(Round) +'.summary'
                if not fio.IsExist(sumfile): break
                Round = Round + 1
            
            Round = Round - 1
            sumfile = oracledir + str(week) + '/' + type + '.' + str(np) + '.L' + str(L) + "." + str(metric) + '.R' + str(Round) +'.summary'
            
            if fio.IsExist(sumfile):
                import os
                ssfile = oracledir + str(week) + '/' + type + '.' + str(np) + '.L' + str(L) + ".summary"
                cmd = 'cp ' + sumfile + ' ' + ssfile
                print cmd
                
                os.system(cmd)
                lines = fio.ReadFile(sumfile)
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
            else:
                row = row + [0]*len(RougeHeader)
            
        body.append(row)
    
    print body
    print "RougeHeader", len(RougeHeader)
    header = ['week'] + RougeHeader*3
    row = []
    row.append("average")
    print len(header)
    for i in range(1, len(header)):
        scores = [float(xx[i]) for xx in body]
        row.append(numpy.mean(scores))
    body.append(row)
    
    fio.WriteMatrix(outputdir + "rouge." + str(np) + '.L' + str(L) + "." + str(metric) + ".txt", body, header)
  
               
def TestRouge():
    #ref = ["police killed the gunman"]
    #S1 = ["police kill the gunman"]
    #S2 = ["the gunman kill police"]
    
    S1_1 = ['Unable to finish the problem set before the quiz',
'Unable to understand why R is useful',
'Some questions were not answered',
'The problem set was confusing',
'Did not understand plots in R, or mean and median'
]
    
    S1_3=['There were two fundamental issues with the class. One, the students felt rushed and were unable to complete the required exercises, and two, the teaching assistant was seen as disorganized and rushed.']
    
    S1_4 = ['The recitation was a good time to learn about the many useful features of R, and how to use them, e.g., plotting data sets and finding the mean and median. The questions in the problem set were well-organized to this end. The application to stock prices was a helpful demonstration.']
    
    S1_2 = ['Learned some useful built-in functions of R',
'Practiced programming in R',
'The question types in the problem set were good',
'Application of R to stock values was helpful',
'The organization of the recitation was good']

    S2_1=['Unfinished In-Class Questions',
'Knowing when to use R',
'Calculating Means/Medians',
'Using / Creating a Histogram',
'Using Rstudio']

    S2_2 = ['Enjoyed Practice Session / In Class Questions',
'Liked learning about capabilities of rstudio',
'Liked Graphical Depictions / Diagrams',
'Liked Learning about Practical Applications of R',
'Enjoyed Calculating R']
    
    S2_3=['Students had trouble knowing when to use R, and would have liked more examples. Some students had trouble using Rstudio. They were also concerned about the questions in the practice session, as they did not get to go over them.']
    
    S2_4 = ['Students seemed to like using rstudio to calculate R, and creating graphics and seeing applications (such as stock values) for R. They liked the practice problems, and would have liked more of them.']
    
    S1_5=['we cannot solve the last two questions of ps if we can solve them before quiz it will be very good ',
'Still not very clear how to use R/what data we were using at all times today when we were discussing R',
'Ps assistant couldn\'t manage her time properly.',
'question 3 in ps 1 was confusing',
'we can plot normal dist when the values are not continous but for it to work we need a lot independent variables , but like how much ?in the question it was 100',
]

    S1_6=['things we can do with r',
    'Question types',
    'I did not think that these questions can be created in lectures.',
    'Interesting to combine the R with statistics on stock values',
    'In the first part of class, we practised R programming, whuch was easy to understand.',
    ]
    
    S2_5=['The r experience was a bit Fast and hard to follow and the ps TA made simple questions look like a bit more complicated',
    'in r function there can be more examples',
    'Still not very clear how to use R/what data we were using at all times today when we were discussing R',
    'inferring informations from histograph is a little confusing since it is new for me',
    'Rstudio ps was not good I couldn\'t concentrate much',
    ]
    
    S2_6=['In the second part, we solved questions, which was helpful for us to remember our probability knowledge.',
    'Most interesting thing was to learn that R is capable of many things like reading Excel or txt files.',
    'Make a result of collection of data, look like normal distribution',
    'the graphics in R and importing data to R',
    'learning how to calculate in R',
    ]

    body = []
    #getRouge(ref, S2)
    r1 = getRouge(S1_1, S2_1)
    r2 = getRouge(S1_2, S2_2)
    r3 = getRouge(S1_3, S2_3)
    r4 = getRouge(S1_4, S2_4)
    r5 = getRouge(S1_5, S2_5)
    r6 = getRouge(S1_6, S2_6)
    
    body.append(r1)
    body.append(r2)
    body.append(r3)
    body.append(r4)
    body.append(r5)
    body.append(r6)
    print body
    
    fio.WriteMatrix('log.txt', body)
    
if __name__ == '__main__':
    oracledir = "../../data/Engineer/Oracle/" 
    datadir = "../../data/Engineer/Oracle/"
    
    #TestRouge()
    
#     for L in [10, 15, 20, 25, 30, 35, 40, 45, 50]:
#         for np in ['syntax', 'chunk']:
#             for metric in ['R1-F', 'R2-F', 'RSU4-F']:
#                 Greedy(oracledir, np, L, metric)
#     
#     for L in [10, 15, 20, 25, 30, 35, 40, 45, 50]:
#         for np in ['syntax', 'chunk']:
#             for metric in ['R1-F', 'R2-F', 'RSU4-F']:
#                 getOracleRouge(oracledir, np, L, metric, datadir)
    
    for L in [30]:
        for np in ['sentence']:
            for metric in ['R2-P', 'R1-P']:
                Greedy(oracledir, np, L, metric)
       
#     for L in [30]:
#         for np in ['sentence']:
#             for metric in ['R2-R']:
#                 getOracleRouge(oracledir, np, L, metric, datadir)
        
    print "done"
    