import Survey
import postProcess
import fio

if __name__ == '__main__':
    
    excelfile = "../../data/2011Spring_norm.xls"
    sennadatadir = "../../data/senna/"
    #outdir = "../../data/ILP_Sentence_Supervised_SVD_Lecture/"
    outdir = "../../data/oracle/"
    
    #Step1: get senna input
    #Survey.getStudentResponses4Senna(excelfile, sennadatadir, Split=False)
    
    #Step2: get senna output
    
    #Step3: get phrases
    #for np in ['syntax', 'chunk']:
    fio.NewPath(outdir)
    
    for np in ['sentence']:
         postProcess.ExtractNPFromRaw(excelfile, sennadatadir, outdir, method=np, Split=True)
         postProcess.ExtractNPSource(excelfile, sennadatadir, outdir, method=np, Split=True)
         postProcess.ExtractNPFromRawWithCount(excelfile, sennadatadir, outdir, method=np, Split=True)
    
    #Step4: write TA's reference 
    Survey.WriteTASummary(excelfile, outdir)
    
    #Greedy()