import tfidf
import fio
import ILP_baseline as ILP
import SennaParser
import porter

from ILP_baseline import stopwords

stopwordfilename = "../../../Fall2014/summarization/ROUGE-1.5.5/data/smart_common_words_stemmed.txt"

tfidfext = ".tfidf"
posext = '.pos'

phraseext = ".key" #a list
studentext = ".keys.source" #json
countext = ".dict"  #a dictionary

def getWordPos(sentence, ngram, NoStopWords=True):
    words_pos = []
    
    tokens = [(porter.getStemming(word.token.lower()), word.pos) for word in sentence.words]
    
    N = len(tokens)
    for n in ngram:
        for i in range(N):
            if i+n > N: continue
            ngram = tokens[i:i+n]
            
            if not NoStopWords:
                words = [w for w,pos in ngram]
                pos = [pos for w,pos in ngram]
                words_pos.append((" ".join(words), ' '.join(pos)))
            else:
                removed = True
                for w,pos in ngram:
                    if w not in stopwords:
                        removed = False
                
                if not removed:
                    words = [w for w,pos in ngram]
                    pos = [pos for w,pos in ngram]
                    words_pos.append((" ".join(words), ' '.join(pos)))
    
    return words_pos

def extact_pos(datadir, sennadatadir, np, ngram):
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
                words_pos = getWordPos(s, ngram)
                
                for w, pos in words_pos:
                    dict[w] = pos
            
            pos_file = prefix + posext
            
            fio.SaveDict(dict, pos_file, SortbyValueflag = True)
                  
if __name__ == '__main__':   
    datadir = "../../data/ILP_Sentence_Supervised_FeatureWeighting/"
    
    sennadatadir = "../../data/senna/"
    
    extact_pos(datadir, sennadatadir, np = 'sentence', ngram=[1,2])