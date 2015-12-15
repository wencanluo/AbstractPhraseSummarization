import Survey
import postProcess
import json
import fio
import NLTKWrapper

phraseext = ".key" #a list
studentext = ".keys.source" #json
countext = ".dict"  #a dictionary
lpext = ".lp"
lpsolext = ".sol"
ngramext = ".ngram.json"

def getNgram(prefix):
    #extract the ngram from the phrase
    data = {}
    
    phrasefile = prefix+phraseext
    
    lines = fio.ReadFile(phrasefile)
    phrases = [line.strip() for line in lines]
    
    #get unigram
    for n in [1,2]:
        ngrams = []
        for phrase in phrases:
            grams = NLTKWrapper.getNgram(phrase, n)
            ngrams = ngrams + grams
        
        ngrams = list(set(ngrams))
        
        data[n] = ngrams
    
    with open(prefix + ngramext, 'w') as outfile:
        json.dump(data, outfile, indent=2)  
        
def ExtractNgram(outdir, np):
    sheets = range(0,12)
    for i, sheet in enumerate(sheets):
        week = i + 1
        dir = outdir + str(week) + '/'
        
        for type in ['POI', 'MP', 'LP']:
            prefix = dir + type + "." + np
            
            getNgram(prefix)

if __name__ == '__main__':    
    excelfile = "../../data/2011Spring_norm.xls"
    sennadatadir = "../../data/senna/"
    
    #fio.NewPath(sennadatadir)
    
    #Step1: get senna input
    #Survey.getStudentResponses4Senna(excelfile, sennadatadir)
    #exit(-1)
    
    #Step2: get senna output
    
    #Step3: get phrases
    for outdir in [
                   #"../../data/ILP_Sentence_Supervised_FeatureWeightingAveragePerceptron/",
                   #"../../data/ILP_Sentence_Supervised_FeatureWeightingAveragePerceptronMC/",
                   #"../../data/ILP_Sentence_Supervised_FeatureWeighting_MC_LCS/",
                   #"../../data/oracle/",
                   #"../../data/ILP_Sentence_Supervised_Oracle/",
                   #"../../data/ILP1_Sentence_MC_Length/",
                   #"../../data/output/conceptweighting/"

                    "../../data/ILP1_Sentence/", 
                    #"../../data/MC/", 
                    #"../../data/ILP1_Sentence_MC_Normalization/", 
                    #"../../data/ILP_Sentence_Supervised_FeatureWeightingAveragePerceptron_Normalization/", 
                    #"../../data/ILP1_Sentence_MC/", 
                    #"../../data/ILP_Sentence_Supervised/",
                    #"../../data/ILP_Sentence_Supervised_FeatureWeighting/", 
                    #"../../data/ILP_Sentence_Supervised_FeatureWeightingMC/", 
                    #"../../data/ILP_Sentence_Supervised_MC/",  
                   ]:
        fio.NewPath(outdir)
          
        for np in ['sentence']:
             postProcess.ExtractNPFromRaw(excelfile, sennadatadir, outdir, method=np, weekrange=range(0,25))
             postProcess.ExtractNPSource(excelfile, sennadatadir, outdir, method=np, weekrange=range(0,25))
             postProcess.ExtractNPFromRawWithCount(excelfile, sennadatadir, outdir, method=np, weekrange=range(0,25))
             #postProcess.ExtractQualityScore(excelfile, sennadatadir, outdir, method=np, weekrange=range(0,12))
        
        #Survey.getStudentResponses4Fei(excelfile, outdir)
        
        #Step4: write TA's reference 
        Survey.WriteTASummary(excelfile, outdir)
    print "done"