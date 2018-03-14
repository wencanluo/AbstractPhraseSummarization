import fio
import json
import sys
import porter
import NLTKWrapper
import os
import json
import numpy
import ILP_baseline as ILP
import global_params

ngramTag = "___"

#Stemming
phraseext = ".key" #a list
studentext = ".keys.source" #json
countext = ".dict"  #a dictionary
lpext = ".lp"
lpsolext = ".sol"

def LoadMC(input):
    with open(input, 'r') as fin:
        A = json.load(fin)
        
    newA = {}
    for bigram, row in A.items():
        bigram = bigram.replace(ngramTag, " ")
        newA[bigram] = row
    return newA

def getNoneZero(A, eps=1e-3):
    nonZero = 0
    N = 0
     
    for bigram, row in A.items():
        N += len(row)
        
        for x in row:
            if x >= eps:
                nonZero = nonZero + 1
    return nonZero, N

def getSparseRatioExample(svddir, prefixA=".org.softA", eps=1e-3):
    sheets = range(0,12)
    
    for sheet in sheets:
        week = sheet + 1
        dir = svddir + str(week) + '/'
        
        for type in ['POI', 'MP', 'LP']:
            svdfile = svddir + str(week) + '/' + type + prefixA
            keyfile = svddir + str(week) + '/' + type + ".sentence.key"
            
            A = LoadMC(svdfile)
            sentences = fio.ReadFile(keyfile)
            
            for bigram, row in A.items():            
                for i, x in enumerate(row):
                    if x >= eps and x != 1.0:
                        print x, '\t', bigram, '@', sentences[i].strip()
    
def getSparseRatio(svddir, prefixA=".org.softA", eps=1e-3):
    sheets = range(0,12)
     
    total_nonZero = 0.0
    total_N = 0.0
    for sheet in sheets:
        week = sheet + 1
        dir = svddir + str(week) + '/'
        
        for type in ['POI', 'MP', 'LP']:
            svdfile = svddir + str(week) + '/' + type + prefixA
        
            A = LoadMC(svdfile)
            
            nonZero, N = getNoneZero(A, eps)
            
            total_nonZero += nonZero
            total_N += N
    
    #print total_nonZero, '\t', total_N, '\t', total_nonZero/total_N
    print eps, '\t', total_nonZero/total_N
    return total_nonZero/total_N
           
def WriteConstraint2(partialBigramPhrase):
    #$\sum_{j=1} {y_j Occ_{ij}} \ge x_i$
    lines = []
    
    for bigram, phrases in partialBigramPhrase.items():
        weightedPhrase = [phrase[1] + ' ' + phrase[0] for phrase in phrases if str(phrase[1]) != '0.0']
        lines.append("  " + " + ".join(weightedPhrase) + " - " + bigram + " >= " + '0')
    return lines

def WriteConstraint3(partialPhraseBigram):
    #$y_j Occ_{ij} \le x_i$
    
    lines = []
    for phrase, bigrams in partialPhraseBigram.items():
        for bigram in bigrams:
            if str(bigram[1].strip()) == '0.0': continue
            lines.append("  " + bigram[1].strip() + ' ' + phrase + " - " + bigram[0] + " <= " + '0')
    return lines
                    
def formulate_problem(BigramTheta, PhraseBeta, partialBigramPhrase, partialPhraseBigram, L, lpfileprefix):
    fio.remove(lpfileprefix + lpext)
    
    lines = []
    
    #write objective
    lines.append("Maximize")
    objective = []
        
    objective = []
    for bigram, theta in BigramTheta.items():
        objective.append(" ".join([str(theta), bigram]))
    lines.append("  " + " + ".join(objective))
    
    #write constraints
    lines.append("Subject To")
    lines += ILP.WriteConstraint1(PhraseBeta, L)
    
    lines += WriteConstraint2(partialBigramPhrase)
    
    lines += WriteConstraint3(partialPhraseBigram)
    
    indicators = []
    for bigram in BigramTheta.keys():
        indicators.append(bigram)
    for phrase in PhraseBeta.keys():
        indicators.append(phrase)
        
    #write Bounds
    lines.append("Bounds")
    for indicator in indicators:
        lines.append("  " + indicator + " <= " + str(1))
    
    #write Integers
    indicators = []
#     for bigram in BigramTheta.keys():
#         indicators.append(bigram)
    for phrase in PhraseBeta.keys():
        indicators.append(phrase)
        
    lines.append("Integers")
    lines.append("  " + " ".join(indicators))
    
    #write End
    lines.append("End")
    fio.SaveList(lines, lpfileprefix + lpext)
    
def getPartialPhraseBigram(IndexPhrase, IndexBigram, phrasefile, svdfile, svdpharefile, threshold):
    lines = fio.ReadFile(phrasefile)
    phrases = [line.strip() for line in lines]
    
    #bigramKeyDict = getRecoveredKeyDict(phrasefile)
    
    lines = fio.ReadFile(svdpharefile)
    svdphrases = [line.strip() for line in lines]
    
    svdA = LoadMC(svdfile)
    
    BigramIndex = {}
    for k,v in IndexBigram.items():
        BigramIndex[v] = k
    
    PhraseIndex = {}
    for k,v in IndexPhrase.items():
        PhraseIndex[v] = k
        
    PartialPhraseBigram = {}
    for phrase in phrases:
        if phrase not in PhraseIndex:
            print "phrase not in PhraseIndex:", phrase
            continue
        i = svdphrases.index(phrase)
        if i==-1:
            print phrase
            continue
        
        pKey = PhraseIndex[phrase]
        if pKey not in PartialPhraseBigram:
            PartialPhraseBigram[pKey] = []
        
        for bigram in svdA.keys():
            row = svdA[bigram]
            svdvalue = row[i]
            
            if bigram not in BigramIndex: 
                #print "bigram not in BigramIndex:"
                continue
            bKey = BigramIndex[bigram]
            
            if row[i] < threshold: continue
            #print bigram, phrase, svdvalue
            #if str(row[i]) == '0.0': continue
            PartialPhraseBigram[pKey].append([bKey, str(row[i])])
    
    PartialBigramPhrase = {}
    for bigram in svdA.keys():       
        row = svdA[bigram]
        
        bigram = bigram.replace(ngramTag, " ")
        if bigram not in BigramIndex: 
            continue
        bKey = BigramIndex[bigram]
        
        if bKey not in PartialBigramPhrase:
            PartialBigramPhrase[bKey] = []
            
        for phrase in phrases:
            if phrase not in PhraseIndex:
                print phrase
                continue
            
            i = svdphrases.index(phrase)
            if i==-1:
                print phrase
                continue
        
            pKey = PhraseIndex[phrase]
            
            if row[i] < threshold: continue
            #if str(row[i]) == '0.0': continue
            PartialBigramPhrase[bKey].append([pKey, str(row[i])])
    
    return PartialPhraseBigram, PartialBigramPhrase
    
def ILP1(prefix, svdfile, svdpharefile, L, Ngram, Lambda, threshold):
    # get each stemmed bigram, sequence the bigram and the phrase
    # bigrams: {index:bigram}, a dictionary of bigram index, X
    # phrases: {index:phrase}, is a dictionary of phrase index, Y
    #PhraseBigram: {phrase, [bigram]}
    IndexPhrase, IndexBigram, PhraseBigram = ILP.getPhraseBigram(prefix + phraseext, Ngram=Ngram, svdfile=svdfile)
    fio.SaveDict(IndexPhrase, prefix + ".phrase_index.dict")
    fio.SaveDict(IndexBigram, prefix + ".bigram_index.dict")
    
    #get weight of bigrams
    BigramTheta = ILP.getBigramWeight_StudentNo(PhraseBigram, IndexPhrase, prefix + countext) # return a dictionary
    fio.SaveDict(BigramTheta, prefix + ".bigram_theta.dict")
    
    #get word count of phrases
    PhraseBeta = ILP.getWordCounts(IndexPhrase)
    
    #getPartial Bigram Phrase matrix
    partialPhraseBigram, PartialBigramPhrase = getPartialPhraseBigram(IndexPhrase, IndexBigram, prefix + phraseext, svdfile, svdpharefile, threshold=threshold)
       
    #get {bigram:[phrase]} dictionary
    BigramPhrase = ILP.getBigramPhrase(PhraseBigram)
    
    lpfile = prefix
    formulate_problem(BigramTheta, PhraseBeta, PartialBigramPhrase, partialPhraseBigram, L, lpfile)
    
    m = ILP.SloveILP(lpfile)
    
    output = prefix + '.L' + str(L) + '.'+str(Lambda)+ '.' + str(threshold) + ".summary"
    print 'output'
    
    ILP.ExtractSummaryfromILP(lpfile, IndexPhrase, output)
    
def ILP_Summarizer(ilpdir, matrix_dir, np, L, Ngram, prefixA, Lambda, threshold, sheets = range(0,12), types=['POI', 'MP', 'LP']):
    
    for sheet in sheets[::-1]:
        week = sheet
        dir = ilpdir + str(week) + '/'
        
        for type in types:
            prefix = dir + type + "." + np
            svdfile = matrix_dir + str(week) + '/' + type + prefixA
            #svdpharefile = matrix_dir + str(week) + '/' + type + '.' + np + ".key"
            svdpharefile = prefix + ".key"
            
            if not fio.IsExist(prefix+phraseext):continue
            
            print prefix
            print svdfile
            
            summary_file = prefix + '.L' + str(L) + '.'+str(Lambda)+ '.' + str(threshold) + ".summary"
            if fio.IsExist(summary_file):
                print 'skip:', summary_file
                continue
            
            ILP1(prefix, svdfile, svdpharefile, L, Ngram, Lambda, threshold=threshold)

def get_ILP_IE256():
    ilpdir = "../../data/IE256/ILP_Sentence_MC/"
    
    from config import ConfigFile
    
    config = ConfigFile(config_file_name='config_IE256.txt')
    
    matrix_dir = "../../data/IE256/MC/"
    print matrix_dir
    
#     A = {'a':[1,0], 'b':[0,0,1]}
#     A = LoadMC("../../data/SVD_Sentence/3/MP.org.softA")
#     print getNoneZero(A)
    
#     matrix_dir = "../../data/matrix/exp5/"
    
    #print getSparseRatio(matrix_dir, prefixA=".500_2.0.softA", eps=0.9)
    #getSparseRatioExample(matrix_dir, prefixA=".500_2.0.softA", eps=0.9)
    #exit(1)
    
    for L in [config.get_length_limit()]:
        for np in ['sentence']:
            rank = config.get_rank_max()
            Lambda = config.get_softImpute_lambda()
            if rank == 0:
                prefixA = '.org.softA'
            else:
                #prefixA = '.' + str(rank) + '_' + str(Lambda) + '.softA'
                prefixA = '.' + str(Lambda) + '.softA'
            
            
            ILP_Summarizer(ilpdir, matrix_dir, np, L, Ngram=config.get_ngrams(), prefixA=prefixA, threshold=config.get_sparse_threshold(), sheets = range(0,26), types=config.get_types()) 
            
    print "done"

def get_ILP_MC_summary(cid, llambda):
    ilpdir = "../../data/%s/ILP_MC/"%cid
    sheets = global_params.lectures[cid]
    print sheets
    
    from config import ConfigFile
      
    config = ConfigFile(config_file_name='config_%s.txt'%cid)
      
    matrix_dir = "../../data/%s/MC/"%cid
    print matrix_dir
    
#     if cid.startswith('CS0445') or cid.startswith('IE256'):
#         LL = [10, 20, 30, 40]
#     elif cid.startswith('Engineer'):
#         LL = [20, 30, 40, 50, 60]
#     elif cid.startswith('review'):
#         LL = [150, 175, 200, 225, 250]
#     else: #news
#         LL = [80, 90, 100, 110, 120]
    
    if cid.startswith('CS0445'):
        LL = [16]
    elif cid.startswith('IE256_2016'):
        LL = [13]
    elif cid.startswith('IE256'):
        LL = [15]
    elif cid.startswith('Engineer'):
        LL = [30]
    elif cid.startswith('review_camera'):
        LL = [216]
    elif cid.startswith('review_IMDB'):
        LL = [242]
    elif cid.startswith('review_prHistory'):
        LL = [190]
    elif cid.startswith('DUC'):
        LL = [105]
    else: #news
        LL = [100]
    
    #for L in [10, 15, 20, 25, 30, 35, 40]:
#     for L in [10, 20, 30, 40]:
#     for L in [20, 30, 40, 50, 60]:
    for L in LL:
#     for L in [80, 90, 100, 110, 120]:
        for np in ['sentence']:
            for Lambda in [llambda]:
            #for Lambda in [0]:
                if Lambda == 0:
                    prefixA = '.org.softA'
                else:
                    prefixA = '.' + str(Lambda) + '.softA'
                  
                ILP_Summarizer(ilpdir, matrix_dir, np, L, Ngram=config.get_ngrams(), prefixA=prefixA, Lambda=Lambda, threshold=config.get_sparse_threshold(), sheets = sheets, types=config.get_types()) 
                  
    print "done"
                
if __name__ == '__main__':
    cid = sys.argv[1]
    llambda = float(sys.argv[2])
    
#     for cid in [
# #                 'IE256',
# #                 'IE256_2016',
# #                 'CS0445',
# #                 'review_camera', 
# #                 'review_IMDB', 
# #                 'review_prHistory',
# #                 'review_all',
# #                 
#                 'DUC04',
#                 'TAC_s08_A',
#                 'TAC_s08_B',
#                 'TAC_s09_A',
#                 'TAC_s09_B',
#                 'TAC_s10_A',
#                 'TAC_s10_B',
#                 'TAC_s11_A',
#                 'TAC_s11_B',
#                 ]:
        
    get_ILP_MC_summary(cid, llambda)
    
    exit(-1)
    
    ilpdir = "../../data/Engineer/ILP_MC/"
     
    from config import ConfigFile
     
    config = ConfigFile()
     
    matrix_dir = config.get_matrix_dir()
    print matrix_dir
     
#     A = {'a':[1,0], 'b':[0,0,1]}
#     A = LoadMC("../../data/SVD_Sentence/3/MP.org.softA")
#     print getNoneZero(A)
     
#     matrix_dir = "../../data/matrix/exp5/"
     
    #print getSparseRatio(matrix_dir, prefixA=".org.softA", eps=1.0)
    getSparseRatioExample(matrix_dir, prefixA=".500_2.5.softA", eps=0.9)
    exit(1)
    
    for L in [config.get_length_limit()]:
        for np in ['sentence']:
            rank = config.get_rank_max()
            Lambda = config.get_softImpute_lambda()
            if rank == 0:
                prefixA = '.org.softA'
            else:
                prefixA = '.' + str(rank) + '_' + str(Lambda) + '.softA'
                #prefixA = '.' + str(Lambda) + '.softA'
             
            print prefixA
             
            ILP_Summarizer(ilpdir, matrix_dir, np, L, Ngram=config.get_ngrams(), prefixA=prefixA, threshold=config.get_sparse_threshold()) 
             
    print "done"