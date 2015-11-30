import fio
import os
import json
import re
import subprocess

#script_filename = 'E:/project/AbstractPhraseSummarization/code/AbstractPhraseSummarization/generate_file.py'
script_filename = 'generate_file.py'#'run_generate.bat' #
script_filename2 = 'run_generate.bat'#'run_generate.bat' #

def create_new_files(datadir, folders):
    for folder in folders:
        path = os.path.join(datadir, folder)
        
        docs = os.path.join(path, 'test_doc_files')
        for subdir, dirs, files in os.walk(docs):
            #if subdir.find('new') == -1: continue
            r = re.compile(r"[a-z][\d]{5}t")
            
            g = r.match(subdir[subdir.rfind('\\')+1:])
            if g:
                print subdir
                
#                 cmd = 'cp %s %s' %(script_filename, subdir)
#                 os.system(cmd)
#                 os.system('chmod 777 %s/%s'%(subdir, script_filename))
#                 
#                 cmd = 'cp %s %s' %(script_filename2, subdir)
#                 os.system(cmd)
#                 
#                 fio.NewPath(os.path.join(subdir, 'new'))
#                 
                #os.chdir(subdir)
                #cmd = 'python %s/%s'%(subdir, script_filename)
                #cmd = 'python %s'%(script_filename)
                #print cmd
                #os.system(cmd)
                
                cmd = 'cmd /C %s/%s/run_generate.bat' %('E:/project/AbstractPhraseSummarization/code/AbstractPhraseSummarization', subdir)
                print cmd
                os.system(cmd)
                
def get_filelist(datadir, folders):
    data = {}
    for folder in folders:
        data[folder] = {}
        
        path = os.path.join(datadir, folder)
        
        docs = os.path.join(path, 'test_doc_files')
        for subdir, dirs, files in os.walk(docs):
            if subdir.find('new') == -1: continue
            
            for file in sorted(files):
                if file.find("py") >= 0:continue
                
                oldname = subdir[:-4]
                doc_id = oldname[oldname.rfind('\\')+1:]
                print doc_id
                
                if doc_id not in data[folder]:
                    data[folder][doc_id] = {'docs':[],'models':[]}
                
                data[folder][doc_id]['docs'].append(os.path.join(subdir, file))

        docs = os.path.join(path, 'models')
        for subdir, dirs, files in os.walk(docs):
            for file in sorted(files):
                #print folder, file
                r = re.compile(r"([A-Z][\d]{5})\.M\.(\d+)\.(.)\..")
                g = r.match(file)
                if g:
                    doc_id = (g.group(1) + g.group(3)).lower()
                    data[folder][doc_id]['models'].append(os.path.join(subdir, file))
                else:
                    print folder, file
    
    fio.SaveDict2Json(data, datadir+'list.json')
            
if __name__ == '__main__':
    datadir = "../../data/DUC04/"
    
    folders = ['DUC_2004']
    
    #create_new_files(datadir, folders)
    get_filelist(datadir, folders)
    