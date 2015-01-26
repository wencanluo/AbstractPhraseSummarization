#!/usr/bin/env python
#file: src.MaximalMatchTokenizer.py
import fio
import NLTKWrapper
import porter

def CheckPhrase(tmp,  phraseset, stemming = True):
    if not stemming:
        return tmp.lower() in phraseset
    
    tmp = tmp.lower()
    tmp = porter.getStemming(tmp)
    if tmp in phraseset:
        return True
    return False

def NgramMatchTokenizer(sentence, phrasefile, maxNgram=5, stemming=True):
    if stemming:
        phraselist = [porter.getStemming(line.lower().strip()) for line in fio.ReadFile(phrasefile)]
    else:
        phraselist = [line.lower().strip() for line in fio.ReadFile(phrasefile)]
    return NgramMatchTokenizerList(sentence, set(phraselist), maxNgram, stemming = stemming)
    
def MaximalMatchTokenizer(sentence, phrasefile, maxNgram=5, stemming=True):
    if stemming:
        phraselist = [porter.getStemming(line.lower().strip()) for line in fio.ReadFile(phrasefile)]
    else:
        phraselist = [line.lower().strip() for line in fio.ReadFile(phrasefile)]
    return MaximalMatchTokenizerList(sentence, set(phraselist), maxNgram, stemming = stemming)

# def ReverseMaximalMatchTokenizer(sentence, phrasefile, maxNgram=5, stemming=True):
#     if stemming:
#         phraselist = [porter.getStemming(line.lower().strip()) for line in fio.ReadFile(phrasefile)]
#     else:
#         phraselist = [line.lower().strip() for line in fio.ReadFile(phrasefile)]
#     
#     return ReverseMaximalMatchTokenizerList(sentence, set(phraselist), maxNgram)

def NgramMatchTokenizerList(sentence, phraseset, maxNgram=5, stemming=True):
    #get unigram list first
    words = NLTKWrapper.getNgram(sentence, 1)
             
    #greedy algorithm
    N = len(words)
    
    k = 0
    
    Order = range(1,maxNgram+1)
    Order.reverse()
    
    phrases = []
    while(k<N):
        founded = False
        for i in Order:
            if k+i > N: continue
            tmp = " ".join(words[k : k+i])
            
            if CheckPhrase(tmp,  phraseset, stemming):
                phrases.append(tmp)
                founded = True
        
        k = k + 1   
    return phrases

def MaximalMatchTokenizerList(sentence, phraseset, maxNgram=5, stemming=True):
    #get unigram list first
    words = NLTKWrapper.getNgram(sentence, 1)
             
    #greedy algorithm
    N = len(words)
    
    k = 0
    
    Order = range(1,maxNgram+1)
    Order.reverse()
    
    phrases = []
    while(k<N):
        founded = False
        for i in Order:
            if k+i > N: continue
            tmp = " ".join(words[k : k+i])
            
            if CheckPhrase(tmp,  phraseset, stemming):
                phrases.append(tmp)
                k = k + i
                founded = True
                break
         
        if not founded:
            k = k + 1   
    return phrases

# def ReverseMaximalMatchTokenizerList(sentence, phraseset, maxNgram=5, stemming=True):
#     #get unigram list first
#     words = NLTKWrapper.getNgram(sentence, 1)
#              
#     #greedy algorithm
#     N = len(words)
#     
#     k = N-1
#     
#     Order = range(maxNgram)
#     Order.reverse()
#     
#     phrases = []
#     while(k>=0):
#         founded = False
#         for i in Order:
#             if k-i < 0: continue
#             tmp = " ".join(words[k-i : k+1])
#             if CheckPhrase(tmp,  phraseset):
#                 phrases.append(tmp)
#                 k = k - i - 1
#                 founded = True
#                 break
#          
#         if not founded:
#             k = k - 1
#     phrases.reverse()
#     return phrases

if __name__ == '__main__':
    ll = ['a', 'a b', 'a b c', 'c d', 'd', 'a b c']
    #ll = ['A', 'B', 'C', 'D']
    #print MaximalMatchTokenizerList("A B C D", ll)
    print NgramMatchTokenizerList("A B C D", ll)
    #print ReverseMaximalMatchTokenizerList("A B C D", ll)