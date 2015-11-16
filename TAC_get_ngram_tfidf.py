import tfidf
import fio
import ILP_baseline as ILP

from ILP_baseline import stopwords

stopwordfilename = "../../../Fall2014/summarization/ROUGE-1.5.5/data/smart_common_words_stemmed.txt"

tfidfext = ".tfidf"
phraseext = ".key" #a list
studentext = ".keys.source" #json
countext = ".dict"  #a dictionary

def getNgrams(prefix, ngram):
    words = []
    PhraseIndex, bigramIndex, PhraseBigram = ILP.getPhraseBigram(prefix + phraseext, Ngram=ngram)
            
    CountFile = prefix + countext
    
    CountDict = fio.LoadDict(CountFile, 'int')

    for phrase, bigrams in PhraseBigram.items():
        assert(phrase in PhraseIndex)
        p = PhraseIndex[phrase]
        try:
            fequency = CountDict[p]
        except Exception as e:
            print p
            exit()
        
        ngrams = [bigramIndex[bigram] for bigram in bigrams]
        
        words += ngrams * fequency
    return words

def extact_tfidf(datadir, np, ngram):
    sheets = range(0,12)
    
    my_tfidf = tfidf.TfIdf(stopword_filename=stopwordfilename)
    
    for i, sheet in enumerate(sheets):
        week = i + 1
        dir = datadir + str(week) + '/'
        
        for type in ['POI', 'MP', 'LP']:
            prefix = dir + type
            prefix = prefix + '.' + np            
            
            words = getNgrams(prefix, ngram)
            my_tfidf.add_input_document_withterms(words)
        
    for i, sheet in enumerate(sheets):
        week = i + 1
        dir = datadir + str(week) + '/'
        
        for type in ['POI', 'MP', 'LP']:
            
            prefix = dir + type
            prefix = prefix + '.' + np  
            words = getNgrams(prefix, ngram)
                                            
            dict = my_tfidf.get_doc_keywords_withterms(words)
            tfidf_file = prefix + tfidfext
            
            fio.SaveDict(dict, tfidf_file, SortbyValueflag = True)
   
if __name__ == '__main__':   
    datadir = "../../data/ILP_Sentence_Supervised_FeatureWeighting/"
    
    extact_tfidf(datadir, np = 'sentence', ngram=[1,2])