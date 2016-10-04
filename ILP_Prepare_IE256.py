import CourseMirror_Survey
import fio
import os

from collections import defaultdict
import global_params

def PrepareCourseForILP(cid):
    excelfile = "../../data/CourseMirror/reflections.json"
    #sennadir = "../data/%s/senna/"%cid
    #fio.NewPath(sennadir)
    #getStudentResponses4Senna(excelfile, cid, maxWeek, sennadir)
    
    outdirs = ['../../data/%s/ILP_Baseline/'%cid,
               '../../data/%s/MC/'%cid,
               '../../data/%s/ILP_MC/'%cid,
               ]
    
    for outdir in outdirs:
        for sheet in global_params.lectures[cid]:
            week = sheet
    
            for type in ['q1', 'q2', 'q3', 'q4']:
                student_summaryList = CourseMirror_Survey.getStudentResponseList(excelfile, cid, week, type, True)
                if len(student_summaryList) == 0: continue
                
                path = os.path.join(outdir, str(week))
                fio.NewPath(path)
                
                source = {}
                responses = []
                count = defaultdict(int)
                for response, student in student_summaryList:
                    responses.append(response)
                    count[response] += 1
                    
                    if response not in source:
                        source[response] = []
                    source[response].append(student)
                    
                outout = os.path.join(path, type + ".sentence.key")
                fio.SaveList(set(responses), outout)
                
                output = os.path.join(path, type + '.sentence.keys.source')
                fio.SaveDict2Json(source, output)
                
                output = os.path.join(path, type + '.sentence.dict')
                fio.SaveDict(count, output)
                                   
if __name__ == '__main__':
#     PrepareCourseForILP('IE256')
#     PrepareCourseForILP('IE256_2016')
    PrepareCourseForILP('CS0445')
    #exit(0)