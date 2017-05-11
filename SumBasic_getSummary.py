import sys
import re
import fio
from collections import defaultdict
from Survey import *
import random
import NLTKWrapper

import SumBasic_word as SumBasic
            
def getShallowSummary(phrasedir, datadir, cid, sheets, types, K):
    #K is the number of words per summary
    for sheet in sheets:
        
        for type in types:
            responses_filename = os.path.join(phrasedir, str(sheet), '%s.sentence.dict'%(type))
            
            student_summaryList = []
            if not fio.IsExist(responses_filename):
                print '%s not exist'%responses_filename
                continue
            
            responses_dict = fio.LoadDict(responses_filename, int)
            
            for sentence, count in responses_dict.items():
                for c in range(count):
                    student_summaryList.append(sentence)

            path = os.path.join(datadir, str(sheet))
            fio.NewPath(path)
            filename = os.path.join(path, type + '.txt')
            
            fio.SaveList(student_summaryList, filename)
            
            #run the SumBasic
            distribution, clean_sentences, processed_sentences = SumBasic.get_sentences(filename)
            summary = SumBasic.summarize(distribution, clean_sentences, processed_sentences, K)
            
            filename = os.path.join(path, type + '.summary')
            fio.SaveList(summary, filename)
            
                        
def ShallowSummary(phrasedir, datadir, cid, sheets, types, K):
    getShallowSummary(phrasedir, datadir, cid, sheets, types, K)
    #WriteTASummary(phrasedir, datadir, cid, sheets, types, K)
        
if __name__ == '__main__':
    from config import ConfigFile
    import global_params
    
    for cid in [
                'Engineer',
                'IE256',
                'IE256_2016',
                'CS0445', 
#                 'review_camera', 
#                 'review_IMDB', 
#                 'review_prHistory',
#                  'DUC04',
                ]:
        config = ConfigFile(config_file_name='config_%s.txt'%cid)
        sheets = global_params.lectures[cid]
        types=config.get_types()
        L = global_params.getLL(cid)[0]
        
        phrasedir = "../../data/"+cid+"/ILP_MC/"
        datadir = "../../data/"+cid+"/SumBasicP/"
        fio.DeleteFolder(datadir)
        
        ShallowSummary(phrasedir, datadir, cid, sheets, types, K=L)

    print 'done'
    