import Survey
from config import ConfigFile

if __name__ == '__main__':
    
    excelfile = "../../data/2011Spring_norm.xls"
    sennadatadir = "../../data/senna/"
    
    config = ConfigFile()
    
    #Step1: get senna input
    Survey.getStudentResponses4Senna(excelfile, sennadatadir, config.get_response_split())
    
    print "getStudentResponses4Senna done"