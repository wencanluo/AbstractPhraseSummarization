import os
import global_params
import fio
import gensim
from ILP_getMatrixCompletion import ProcessLine
import collections
from exp_select_lambda import get_ttest_pvalues
import numpy

phrase_exe = '.key'
color_exe  = '.key.color'

pos_tag = '+'
neg_tag = '-'
color_tag = '@color@'
h1_tag = '@H1@'
h2_tag = '@H2@'

class EvalMC:
    def __init__(self, phrasedir, sentencedir, cid):
        self.phrasedir = phrasedir
        self.sentencedir = sentencedir
        
        self.cid = cid
        self.phrasefolder = os.path.join(phrasedir, cid)
        self.sentencefolder = os.path.join(sentencedir, cid)
        self.N = 2
    
    def combine_phrase_color(self, phrases, color_map):
        phrase_color_map = {}
        
        assert(len(phrases) == len(color_map))
        
        for phrase, color_list in zip(phrases, color_map):
            if phrase not in phrase_color_map:
                phrase_color_map[phrase] = [[] for i in range(self.N)]
            
            for i,colors in enumerate(color_list):
                phrase_color_map[phrase][i] = phrase_color_map[phrase][i] + colors
        
        return phrase_color_map
    
    def similar_color(self, color1, color2):
        n = len(color1)
        for i in range(n):
            for c1 in color1[i]:
                if c1 == -1: continue
                if c1 in color2[i]:
                    return True
        
    def simliar(self, b1, b2, D):
        assert(b1 in D)
        assert(b2 in D)
        
        color1 = D[b1]['color']
        color2 = D[b2]['color']
        
        return self.similar_color(color1, color2)
    
    def extract_bigram_number(self):
        method = 'crf'
        sheets = global_params.Phraselectures[self.cid]
        ngrams = [2]
        
        data = {}
        
        count, simcount, diffcount, H1_c, H2_c = 0, 0, 0, 0, 0
        for lec in sheets:
            data[lec] = {}
            
            for prompt in ['q1', 'q2']:
                dict = {} #save bigram, its phrase, its sentence id, and colors
                
                for annotator in ['1', '2']:
                    key_prefix = os.path.join(self.phrasefolder, 'oracle_annotator_%s'%annotator, 'phrase', str(lec), '%s.%s'%(prompt, method))
                    
                    #load phrase color map
                    phrasefile = key_prefix + phrase_exe
                    phrases = fio.LoadList(phrasefile)
                    
                    colorfile = key_prefix + color_exe
                    color_map = fio.LoadDictJson(colorfile)
                    
                    #extract the color of the phrases
                    phrase_color_map = self.combine_phrase_color(phrases, color_map)
                    
                    #extract bigrams from phrases, and mark the color
                    for phrase in phrases:
                        line = ProcessLine(phrase, ngrams)
                        bigrams = list(gensim.utils.tokenize(line, lower=True, errors='ignore'))
                        
                        if len(bigrams) == 1:
                            #print bigrams
                            count += 1
                            
                            bigram = bigrams[0]
                            
                            if bigram not in dict: 
                                dict[bigram] = {}
                                dict[bigram]['phrase'] = []
                                dict[bigram]['color'] = [[],[]]
                                dict[bigram][pos_tag] = []
                                dict[bigram][neg_tag] = []
                            
                            dict[bigram]['phrase'].append(phrase)
                            
                            colors = phrase_color_map[phrase]
                            
                            for i, color in enumerate(colors):
                                dict[bigram]['color'][i] += color
                            
                #extract sentence id of the bigram
                #read sentences, and save its index
                sentence_file = os.path.join(self.sentencefolder, 'MC', str(lec), '%s.sentence.key'%(prompt))
                sentences = [' '+line.lower()+' ' for line in fio.LoadList(sentence_file)]
                
                removed_bigrams = []
                for bigram in dict:
                    dict[bigram]['sentence'] = []
                    for phrase in set(dict[bigram]['phrase']):
                        found = False
                        for sid, sentence in enumerate(sentences):
                            if sentence.find(' '+phrase+' ') != -1:
                                found = True
                                dict[bigram]['sentence'].append(sid)
                        
                        if not found:
                            #print cid, lec, prompt, bigram, dict[bigram]
                            removed_bigrams.append(bigram)
                
                for bigram in set(removed_bigrams):
                    del dict[bigram]
                
                #extract sentence color
                s_colors = []
                for i, sentence in enumerate(sentences):
                    tcolors = [[],[]]
                    for phrase in phrases:
                        if sentence.find(' '+phrase+' ') != -1:
                            colors = phrase_color_map[phrase]
                            for i, color in enumerate(colors):
                                tcolors[i] += color
                     
                    for i in range(len(tcolors)):
                        x = set(tcolors[i])
                        x.discard(-1)
                        tcolors[i] = list(x)
                    s_colors.append(tcolors)
                
                #simplify the dict
                for bigram in dict:
                    dict[bigram]['phrase'] = list(set(dict[bigram]['phrase']))
                    for cii in range(len(dict[bigram]['color'])):
                        dict[bigram]['color'][cii] = list(set(dict[bigram]['color'][cii]))
                    dict[bigram]['sentence'] = list(set(dict[bigram]['sentence']))
                
                count += len(dict)
                    
                #extract similar bigrams and different bigrams
                bigrams = dict.keys()
                n = len(bigrams)
                
                dict[color_tag] = s_colors
                
                if n > 1:
                    for i in range(n):
                        for j in range(i+1, n):
                            b1 = bigrams[i]
                            b2 = bigrams[j]
                            if self.simliar(b1, b2, dict):
                                dict[b1][pos_tag].append(b2)
                                dict[b2][pos_tag].append(b1)
                                
                                simcount += 2
                            else:
                                dict[b1][neg_tag].append(b2)
                                dict[b2][neg_tag].append(b1)
                                
                                diffcount += 2
                
                H1_selected = []
                
                #for H1: select B, S1, S2 (S1 has B1 = B, S2 has B2 != B and not B3=B) 
                for B in bigrams:
                    if len(dict[B]) == 0: continue
                    
                    color_b = dict[B]['color'] 
                    
                    for B1 in dict[B][pos_tag]:
                        #print B, B1
                        for S1 in dict[B1]['sentence']:
                            
                            foundS2 = False
                            for B2 in dict[B][neg_tag]:
                                for S2 in dict[B2]['sentence']:
                                    #check the color
                                    s2_color = s_colors[S2]
                                    if self.similar_color(s2_color, color_b): continue
                                    
                                    foundS2 = True
                                    break
                                
                                if foundS2: break
                            
                            if foundS2:
                                #print 'H1', B, S1, S2, B1, B2
                                H1_selected.append((B, S1, S2, B1, B2))
                
                H1_c += len(H1_selected)
                
                dict[h1_tag] = H1_selected
                
                #for H2: select S, B1, B2 (S has B = B1, S not have B = B2)
                H2_selected = []
                for B in bigrams:
                    if len(dict[B]) == 0: continue
                    
                    for S in dict[B]['sentence']: 
                        s_color = s_colors[S]
                        
                        for B1 in dict[B][pos_tag]:
                            for B2 in dict[B][neg_tag]:
                                b2_color = dict[B]['color']
                                
                                if self.similar_color(s_color, b2_color): continue
                                
                                #print 'H2', S, B1, B2, B
                                H2_selected.append((S, B1, B2, B))
                
                H2_c += len(H2_selected)
                dict[h2_tag] = H2_selected
                
                data[lec][prompt] = dict
        
        output = os.path.join(self.sentencedir, cid, 'MC', 'MC_bigram.json')
        #print output
        
        fio.SaveDict2Json(data, output)
        
        return [count, simcount, diffcount, H1_c, H2_c]

def H(cid, sentencedir, L=10, h_tag=h1_tag):
    folder = os.path.join(sentencedir, cid, 'MC') 
    mc_bigram = fio.LoadDictJson(os.path.join(folder, 'MC_bigram.json'))
    lambda_file = fio.LoadDictJson(os.path.join(folder, 'lambda_%d.json'%L))
    
    sheets = global_params.Phraselectures[cid]
    
    body1 = []
    body2 = []
    
    for lec in sheets:
        for prompt in ['q1', 'q2']:
            D = mc_bigram[str(lec)][prompt]
            lambda_p = lambda_file[str(lec)][0]
            
            mc_matrix_file = os.path.join(folder, str(lec), '%s.%s.softA'%(prompt, lambda_p))
            
            A = fio.LoadDictJson(mc_matrix_file)
            
            if h_tag == h2_tag:
                for S, B1, B2, B in D[h2_tag]:
                    body1.append([A[B1][S]])
                    body2.append([A[B2][S]])
            elif h_tag == h1_tag:
                for B, S1, S2, B1, B2 in D[h1_tag]:
                    body1.append([A[B][S1]])
                    body2.append([A[B][S2]])
    
    pvalues = get_ttest_pvalues(body1, body2, [0])
    
    m1, m2, p = numpy.mean([row[0] for row in body1]), numpy.mean([row[0] for row in body2]), pvalues[0]
    bigger = True if m1 > m2 else False
    
    m1 = '%.3f'%m1
    m2 = '%.3f'%m2
    
    if p < 0.05:
        if bigger:
            m1 = m1 + '+'
        else:
            m1 = m1 + '-'
    
    return m1, m2, '%.3f'%p

if __name__ == '__main__':
    phrasedir = '../../data/'
    sentencedir = '../../data/'
    
    for cid in ['IE256_nocutoff',
                'IE256_2016_nocutoff',
                'CS0445_nocutoff',
                ]:
        print cid
        
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
            
        for L in LL:
#             eval_mc = EvalMC(phrasedir, sentencedir, cid)
#             print '\t'.join([str(x) for x in eval_mc.extract_bigram_number()])
#             
            print '\t'.join(H(cid, sentencedir, L, h1_tag))
#         
#         for L in [10, 20, 30, 40]:
            print '\t'.join(H(cid, sentencedir, L, h2_tag))
        
        