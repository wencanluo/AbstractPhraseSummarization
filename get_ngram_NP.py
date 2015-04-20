import tfidf
import fio
import ILP_baseline as ILP
import SennaParser
import porter
import get_ngram_pos

from ILP_baseline import stopwords

stopwordfilename = "../../../Fall2014/summarization/ROUGE-1.5.5/data/smart_common_words_stemmed.txt"

tfidfext = ".tfidf"
posext = '.pos'
inNPext = '.inNP'

phraseext = ".key" #a list
studentext = ".keys.source" #json
countext = ".dict"  #a dictionary

def extact_inNP(datadir, sennadatadir, np, ngram):
    sheets = range(0,12)
    
    for i, sheet in enumerate(sheets):
        week = i + 1
        dir = datadir + str(week) + '/'
        
        for type in ['POI', 'MP', 'LP']:
            prefix = dir + type
            prefix = prefix + '.' + np            
            
            dict = {}
            
            sennafile = sennadatadir + "senna." + str(week) + "." + type + '.output'
            
            sentences = SennaParser.SennaParse(sennafile)
            for s in sentences:
                words_pos = get_ngram_pos.getWordPos(s, ngram)
                
                NPs = s.getSyntaxNP()
                NPs = [' ' + porter.getStemming(NP) + ' ' for NP in NPs]
                
                for w, pos in words_pos:
                    tw = ' ' + w + ' '
                    
                    for NP in NPs:
                        if NP.find(tw) != -1:
                            dict[w] = 1
                            break
            
            np_file = prefix + inNPext
            
            fio.SaveDict(dict, np_file, SortbyValueflag = True)
                  
if __name__ == '__main__':   
    datadir = "../../data/ILP_Sentence_Supervised_FeatureWeighting/"
    
    sennadatadir = "../../data/senna/"
    
    extact_inNP(datadir, sennadatadir, np = 'sentence', ngram=[1,2])