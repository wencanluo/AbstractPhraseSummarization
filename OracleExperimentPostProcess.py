import postProcess
    
if __name__ == '__main__':
    oracledir = "../../data/oracle/" 
    
    models = []
    for np in ['syntax', 'chunk']:
        for metric in ['R1-F', 'R2-F', 'RSU4-F']:
            for L in [10, 15, 20, 25, 30, 35, 40, 45, 50]:
                model = str(np) + '.L' + str(L) + "." + str(metric)
                models.append(model)
    
    postProcess.CombineRouges(models, oracledir)
            
    print "done"
    