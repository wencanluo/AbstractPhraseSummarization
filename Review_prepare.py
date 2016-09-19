import fio
import os
import collections
import NLTKWrapper
import numpy

def SaveReview(filename, outputdir):
    lines = fio.LoadList(filename)
    
    k = 0
    
    responses = []
    source = collections.defaultdict(list)
    count = collections.defaultdict(int) 
    
    while k < len(lines):
        id_line = lines[k]
        id = id_line.split(':')[-1].strip()
        
        text_line = lines[k+1]
        texts = NLTKWrapper.splitSentence(text_line)
        sentences = [s.strip() for s in texts]
        responses += sentences
        
        for s in sentences:
            source[s].append(id)
            count[s] += 1
        
        k += 3
    
    outout = outputdir + ".sentence.key"
    fio.SaveList(set(responses), outout)
    
    output = outputdir + '.sentence.keys.source'
    fio.SaveDict2Json(source, output)
    
    output = outputdir + '.sentence.dict'
    fio.SaveDict(count, output)
    
    
def perpare_review(sumdir, system):
    datadir = '../../data/review/thesis_summarization_datasets/'
    
    #prepare text
    #iter the folder
    text_dir = os.path.join(datadir, 'review_text')
    ref_dir  = os.path.join(datadir, 'autoeval/refs/')
    
    D = collections.defaultdict(list)
    
    for subdir, dirs, files in os.walk(text_dir):
        for filename in files:
            tokens = filename.split('.')
            ext = tokens[-1]
            name = '.'.join(tokens[:-1])
            domain = name.split('_')[0]
            
            D[domain].append(name)
    
    #extract text
    for domain in D:
        #create a folder
        path = os.path.join(sumdir, 'review_%s'%domain, '%s'%system) 
        fio.NewPath(path)
        
        counts = []
        for lec, name in enumerate(D[domain]):
            folder = os.path.join(path, str(lec))
            
#             fio.NewPath(folder)
#             
#             prefix = os.path.join(folder, 'q1')
#             filename = os.path.join(text_dir, '%s.reviews'%name)
#             print filename
#             
#             SaveReview(filename, prefix)
                        
            ref_folder = os.path.join(ref_dir, name)
            
            #iter all human summaries
            for subdir, dirs, files in os.walk(ref_folder):
                for rid, refname in enumerate(files):
                    lines = fio.LoadList(os.path.join(subdir, refname))
                    
                    sumfile = os.path.join(folder, 'q1.ref.%d'%rid)
                    #fio.SaveList(lines, sumfile)
                     
                    wc = 0
                    for line in lines:
                        wc += len(line.split())
                    counts.append(wc)
        
        print domain, counts, numpy.mean(counts)
    
if __name__ == '__main__':
    
    datadir = '../../data/'
    systems = ['ILP_baseline', 'ILP_MC',  'MC']
    for system in systems:
        perpare_review(datadir, system)
    