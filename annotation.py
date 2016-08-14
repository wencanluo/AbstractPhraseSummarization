# -*- coding: utf-8 -*-
import fio
import re
import json
from _codecs import decode
from numpy import rank

from global_params import g_cid

if g_cid == 'IE256':
    doc_prefix = '_IE256_Lecture_'
    Lectures = [x for x in range(14, 26) if x != 22]
    AllLectures = [x for x in range(3, 26) if x != 22]
    PrefrenceLectures = [3, 4] + [x for x in range(10, 26) if x != 22]
    
    datadir = "../data/IE256/"
    anotators = ['Youngmin', 'Trevor']
    anotator_dict = {'Youngmin':0, 
                     'Trevor':1}

elif g_cid == 'IE256_2016':
    doc_prefix = '_Lecture_'
    Lectures = [x for x in range(3, 27)]
    AllLectures = [x for x in range(3, 27)]
    PrefrenceLectures = [x for x in range(3, 27)]
    
    datadir = "../data/IE256_2016/"
    anotators = ['Waiwood', 'Wang']
    anotator_dict = {'Waiwood':0, 
                     'Wang':1}
    
elif g_cid == 'CS0445':
    doc_prefix = '_Lecture_'
    Lectures = [x for x in range(5, 29) if x != 15]
    AllLectures = Lectures
    PrefrenceLectures = Lectures
    
    datadir = "../data/CS0445/"
    anotators = ['Sam', 'Dana']
    anotator_dict = {'Sam':0, 
                     'Dana':1}

summarization_methods = ['Phrase', 'Abstract', 'Extractive']

prompt_dict = {'q1':0,
               'q2':1,
               '0':0,
               '1':1,
               'POI':0,
               'MP':1,
               }

prompt_words = {
                'q1': 'describe what you found most interesting in today\'s class'.split(),
                'q2': 'describe what was confusing or needed more detail'.split(),
                }

prompt_name = {'q1':'POI',
               'q2':'MP'
               }
               
def get_name(lec, anotator):
    return anotator + doc_prefix + str(lec) + '_Completed'
    
def generate_all_files(datadir, extension, anotators=anotators, lectures = AllLectures):
    for annotator in anotators:
        for lec in lectures:
            filename = datadir + annotator + doc_prefix + str(lec) + '_Completed' + extension
            
            assert(fio.IsExist(filename))
            
            yield filename, lec, annotator

def generate_all_files_by_annotators(datadir, extension, anotators=anotators, lectures = AllLectures):
    for lec in lectures:
        docs = []
        for annotator in anotators:
            filename = datadir + annotator + doc_prefix + str(lec) + '_Completed' + extension
            assert(fio.IsExist(filename))
            
            docs.append( (filename, lec, annotator) )
        yield docs
                        
class Task:
    def __init__(self):
        pass
    
    def extract_start_time(self):
        value = []
        for line in self.lines:
            g = re.match('Start Time:\s*_*(\d[^_]*)_+', line)
            if g: #start time
                value.append(g.group(1))
        
        assert(len(value) == 2)
        self.start_time = value
    
    def get_start_time(self):
        return self.start_time
    
    def extract_finish_time(self):
        value = []
        for line in self.lines:
            g = re.match('Finish Time:\s*_*(\d[^_]*)_+', line)
            if g: #start time
                value.append(g.group(1))
        
        if len(value) != 6:
            print "extract_finish_time", self.filename
            
        assert(len(value) == 6)
        self.finish_time = value
    
    def get_finish_time(self):
        return self.finish_time
    
    def extract_prompts(self):
        regex = re.compile("Prompt\d: “(.*)”", re.UNICODE)
        
        value = []
        for line in self.lines:
            g = regex.match(line)
            if g: #start time
                value.append(g.group(1))
        
        if len(value) != 2:
            print "extract_prompts", self.filename
            
        assert(len(value) == 2)
        self.prompts = value
    
    def get_prompts(self):
        return self.prompts
    
    def extract_responses(self):
        regex = re.compile("Responses from students")
        
        self.responses = []
        
        body = []
        k = 0
        N = len(self.lines)
        
        while(k < N):
            line = self.lines[k]
            k += 1
            
            g = regex.match(line)
            if g: break
        
        #start to read the response
        state = 0
        body = []
        row = []
        
        while(k < N):
            line = self.lines[k]
            k += 1
            
            if line.startswith('Task1'): #end of the first response
                break
            
            line = line.rstrip("\r\n")
            
            if len(line) == 0:
                state += 1
                if state == 4: state = 0
                continue
            
            row.append(line)
              
            if state == 3:
                if len(row) != 3:
                    print "extract_responses", self.filename, row
                body.append(row)
                row = []
                state = 0
        
        self.responses.append(body) #first prompt
        
        
        while(k < N):
            line = self.lines[k]
            k += 1
            
            g = regex.match(line)
            if g: break
        
        #start to read the response
        state = 0
        body = []
        row = []
        
        while(k < N):
            line = self.lines[k]
            k += 1
            
            if line.startswith('Task1'): #end of the second response
                break
            
            line = line.rstrip("\r\n")
            
            if len(line) == 0:
                state += 1
                if state == 4: state = 0
                continue
            
            row.append(line)
              
            if state == 3:
                if len(row) != 3:
                    print "extract_responses2", self.filename, row
                    
                body.append(row)
                row = []
                state = 0
        
        self.responses.append(body) #first prompt
    
    def get_responses(self):
        return self.responses
     
    def extract_task_names(self):
        regex = re.compile('Task\d: (Extractive|Abstract|Phrase) (summary|Summarization)')
        
        value = []
        for line in self.lines:
            g = regex.search(line)
            if g:
                value.append(g.group(1))
        
        if len(value) != 6:
            print "extract_task_names", self.filename
            
        assert(len(value) == 6)
        self.task_names = value
    
    def get_task_names(self):
        self.task_names = []
        for task in self.tasks:
            self.task_names.append(task['task_name'])
        return self.task_names 
    
    def extract_task_anntation(self):
        regex_begin = re.compile('Task\d: (Extractive|Abstract|Phrase) (summary|Summarization)')
        regex_finish = re.compile('Finish Time:\s*_*(\d[^_]*)_+')
        regex_rank = re.compile('Rank\d: _*(\d+)_+')
        regex_abstract_begin = re.compile('^%type your summary below$')
        regex_abstract_end1 = regex_abstract_begin
        regex_abstract_end2 = regex_finish
        regex_abstract_end3 = regex_begin
        
        self.tasks = []
        
        state = 0
        value = []
        for line in self.lines:
            line = line.rstrip()
            
            g = regex_begin.search(line)
            if g: 
                state = 1
                task = g.group(1)
                dict = {}
                dict['task_name'] = task
                sub_task_state = 0
                continue
            
            g = regex_finish.search(line)
            if g: 
                self.tasks.append(dict)
                
                state = 0
                continue

            if state == 0: continue
            
            if task == 'Extractive':
                g = regex_rank.search(line)
                if g:
                    if 'summary' not in dict: dict['summary'] = []
                    dict['summary'].append(g.group(1))
            
            elif task == 'Abstract':
                if sub_task_state == 1:
                    if regex_abstract_end1.search(line) or regex_abstract_end2.search(line) or regex_abstract_end3.search(line):
                         sub_task_state = 0
                         continue
                    else:
                        if len(line) != 0:
                            if 'summary' not in dict: dict['summary'] = []
                            dict['summary'].append(line)
                
                g = regex_abstract_begin.search(line)
                if g:#state
                    sub_task_state = 1
                    continue
            
            elif task == 'Phrase':
                if sub_task_state == 0:
                    row = []
                    
                if len(line) == 0:
                    sub_task_state += 1
                    if sub_task_state == 4: sub_task_state = 0
                    continue
                
                if line.startswith('Note, please also highlight the corresponding'):
                    sub_task_state = 0
                    continue
                
                row.append(line)
                  
                if sub_task_state == 3:
                    if 'summary' not in dict: dict['summary'] = []
                    
                    if len(row) != 3:
                        print "extract_task_anntation", self.filename
                        
                    dict['summary'].append(row)
                    row = []
                    sub_task_state = 0
        
        if len(self.tasks) != 6:
            print "extract_task_anntation", self.filename
    
    def extract_preferece(self):
        regex = re.compile('_*(\d[^_]*)_+$')
                 
        values = []
        state = 0
        for line in self.lines:
            if line.startswith('If you had to choose only one summary among the task'):
                state = 1
                continue
            
            if state == 1:  
                g = regex.match(line)
                if g:
                    values.append(g.group(1))
        
        if len(values) < 1:
            print "extract_preferece", self.filename
                    
        if len(values) == 1:
            values.append(values[0])
        
        self.preference = []
        for v in values:
            self.preference.append(self.task_names[int(v)])
        
    def get_tasks(self):
        return self.tasks
        
    def combine_info(self):
        self.extract_start_time()
        self.extract_finish_time()
        self.extract_prompts()
        self.extract_responses()
        self.extract_task_names()   
        self.extract_preferece()
        self.extract_raw_response()
        self.extract_phrase_annotation()
        
        for i, task in enumerate(self.tasks):
           task['finish_time'] = self.finish_time[i]
           task['prompt'] = 0 if i < 3 else 1
           task['response'] = 0 if i < 3 else 1
           
           if i == 0 or i==3:
               task['start_time'] = self.start_time[(i+1)/3]
           else:
               task['start_time'] = self.finish_time[i-1] 
        
        self.data = {}
        self.data['tasks'] = self.tasks
        self.data['prompts'] = self.prompts
        self.data['responses'] = self.responses
        self.data['start_time'] = self.start_time
        self.data['preference'] = self.preference
        self.data['raw_response'] = self.raw_response
        self.data['phrase_annotation'] = self.phrase_annotation
    
    def load(self, filename):
        self.filename = filename
        self.lines = fio.ReadFile(filename)
        
        self.extract_task_anntation()
        
        self.combine_info()

    def save2json(self, out):
        with open(out, 'w') as fout:
            json.dump(self.data, fout, indent=2, encoding='utf-8')
    
    def loadjson(self, jsonfile):
        self.filename = jsonfile
        with open(jsonfile, 'r') as fin:
            self.data = json.load(fin, encoding='utf-8')
        
        self.tasks = self.data['tasks']
        self.prompts = self.data['prompts']
        self.responses = self.data['responses']
        self.start_time = self.data['start_time']
        self.preference = self.data['preference']
        self.raw_response = self.data['raw_response']
        self.phrase_annotation = self.data['phrase_annotation']
        
    def normalize(self, response):
        return re.sub(r'<(.*?)><(\d)>', r'\1', response)
        
    def extract_raw_response(self):
        self.raw_response = []
        for response in self.responses:
            raw_response = []

            for student, id, sentence in response[1:]:
                dict = {'student_id': student.strip(),
                        'sentence_id': id.strip(),
                        'response' : self.normalize(sentence.strip())
                        }
            
                raw_response.append(dict)
            self.raw_response.append(raw_response)
    
    def get_raw_response(self, prompt):
        return self.raw_response[prompt_dict[prompt]]
    
    def get_response(self, prompt):
        return self.responses[prompt_dict[prompt]]
    
    def get_phrase_summary_textdict(self, prompt):
        phrases = self.get_phrase_summary(prompt)
        
        pattern = re.compile('<(\d+)>')
        dict = {}
        for phrase in phrases:
            rank, text, student_number = phrase
            g = re.search(pattern, rank)
            rank = g.group(1)
            dict[rank] = text
            
        return dict    
            
    def get_phrase_summary_student_coverage(self, prompt):
        phrases = self.get_phrase_summary(prompt)
        
        pattern = re.compile('<(\d+)>')
        dict = {}
        for phrase in phrases:
            rank, text, student_number = phrase
            g = re.search(pattern, rank)
            rank = g.group(1)
            dict[rank] = int(student_number)
            
        return dict
        
    def get_phrase_summary(self, prompt, withhead=False):
        sub_tasks = self.get_tasks()
        
        for sub_task in sub_tasks:
            if sub_task["task_name"] == "Phrase":
                if sub_task['prompt'] == prompt_dict[prompt]: #POI
                    if withhead:
                        return sub_task["summary"][0], sub_task["summary"][1:]
                    else:
                        return sub_task["summary"][1:]
        
        if withhead:
            return None, None
        else:
            return None
    
    def get_phrase_annotation(self, prompt):
        return self.phrase_annotation[prompt_dict[prompt]]
        
    def extract_phrase_annotation(self):
        regex = re.compile("<(.*?)><(\d)>")
        
        self.phrase_annotation = []
        for response in self.responses:
            raw_response = []
            
            dict = {}
            for student, _, sentence in response[1:]:
                gs = regex.findall(sentence)
               
                for g in gs:
                    rank = g[1]
                    phrase = g[0]
                    if rank not in dict:
                        dict[rank] = []
                    
                    dict[rank].append({'student_id': student,
                                       'phrase': phrase,
                                       })
            
            self.phrase_annotation.append(dict)
    
    def get_number_of_sentences(self):
        N = [0]*len(self.raw_response)
        for i, responses in enumerate(self.raw_response):
            for dict in responses:
                N[i] += 1
        
        self.number_of_sentences = N
        return N
    
    def get_students(self):
        V = [set()]*len(self.raw_response)
        for i, responses in enumerate(self.raw_response):
            for dict in responses:
                if len(dict['student_id'].strip()) == 0: continue
                V[i].add(dict['student_id'])
                
        self.students = V
        
        return V
    
    def get_number_of_words(self):
        N = [0]*len(self.raw_response)
        for i, responses in enumerate(self.raw_response):
            for dict in responses:
                response = dict['response']
                response = response.strip()
                dict['number_of_words'] = len(response.split())
                N[i] += len(response.split())
        
        self.number_of_words = N
        return N
    
    def get_summary_text(self, task):
        #return a list of text
        summaries = []
        
        if task['task_name'] == 'Extractive':
            responses = self.raw_response[task['response']]
            
            for sentence_id in task['summary']:
                student_response = responses[int(sentence_id)-1]
                if sentence_id != student_response['sentence_id']:
                    print "get_summary_text", self.filename, sentence_id, student_response, 
                assert(sentence_id == student_response['sentence_id'])
                sentence = student_response['response']
                summaries.append(sentence)
        elif task['task_name'] == 'Abstract':
            summaries = task['summary']
            
        elif task['task_name'] == 'Phrase':
            for rank, phrase, coverage in task['summary'][1:]:
                summaries.append(phrase)
                
        return summaries
        
    def get_summary_length(self):
        N = [0]*len(self.tasks)
        for i, task in enumerate(self.tasks):
            summaries = self.get_summary_text(task)
            
            for summary in summaries:
                N[i] += len(summary.split())
        
        self.summary_length = N
        return N
    
    def get_task_times(self):
        import util

        N = [0]*len(self.tasks)
        for i, task in enumerate(self.tasks):
            
            dt = util.datatime2seconds(util.str2dt(task['finish_time'])) - \
                util.datatime2seconds(util.str2dt(task['start_time']))
            
            if dt < 0:
                print "get_task_times", self.filename
                
            N[i] = dt
        
        self.task_times = N
        return N
    
    def get_preference(self):
        return self.preference
    
    def sort_by_name(self, unsorted):
        assert(len(unsorted) == len(summarization_methods) or len(unsorted) == len(summarization_methods)*2)
        
        sorted = [0]* len(unsorted) 
        task_names = self.get_task_names()
        
        for i, x in enumerate(unsorted):
            index = summarization_methods.index(task_names[i])
            sorted[index + i/3*3] = x
            
        return sorted
        
if __name__ == '__main__':
    
    task = Task()
    
    task.load("../data/CS0445/text/Dana_Lecture_5_Completed.txt")
    
    #task.loadjson("../data/IE256/json/Trevor_IE256_Lecture_14_Completed.json")
    #print task.normalize("<probability of making mistake><2> is a <little><1> confusing")
    
    #print task.get_number_of_sentences()
    #print task.get_students()
    #print task.get_number_of_words()
    #print task.get_task_times()
    
    print task.get_phrase_summary_textdict('q1')
    
#     print summarization_methods
#     print task.sort_by_name(task.get_task_times())
    
    