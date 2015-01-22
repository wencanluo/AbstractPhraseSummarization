import Survey
import postProcess

if __name__ == '__main__':
    
    excelfile = "../../data/2011Spring_norm.xls"
    sennadatadir = "../../data/senna/"
    outdir = "../../data/ILP/"
    
    #Step1: get senna input
    #Survey.getStudentResponses4Senna(excelfile, sennadatadir)
    
    #Step2: get senna output
    
    #Step3: get phrases
#     for np in ['syntax', 'chunk']:
#          postProcess.ExtractNPFromRaw(excelfile, sennadatadir, outdir, method=np)
#          postProcess.ExtractNPSource(excelfile, sennadatadir, outdir, method=np)
#          postProcess.ExtractNPFromRawWithCount(excelfile, sennadatadir, outdir, method=np)
    
    #Step4: write TA's reference 
    Survey.WriteTASummary(excelfile, outdir)
    
    #Greedy()