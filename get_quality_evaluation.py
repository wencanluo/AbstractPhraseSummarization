import fio, numpy

def getQuality(datadir, np, L, outputdir, Lambda):
    sheets = range(0,12)
    
    body = []
    
    for i, sheet in enumerate(sheets):
        week = i + 1
        
        for type in ['MP']:
            row = []
            row.append(week)
        
            sumfile = datadir + str(week) + '/' + type + '.' + str(np) + '.L' + str(L) + '.summary'
            
            lines = fio.ReadFile(sumfile)
            TmpSum = [line.strip() for line in lines]
            
            score = []
            
            quality_file = datadir + str(week)+ '/' + type + '.'+np+'.quality'
            if fio.IsExist(quality_file):            
                dict = fio.LoadDict(quality_file)
   
                for s in TmpSum:
                    if s in dict:
                        ss = dict[s]
                        if ss == 'a': continue
                        score.append(float(ss))
                
            if len(score) > 0:
                row = row + [numpy.sum(score)]
            else:
                row = row + [0]
            
            body.append(row)
            
    header = ['week'] + ['quality']    
    row = []
    row.append("average")
    for i in range(1, len(header)):
        scores = [float(xx[i]) for xx in body]
        row.append(numpy.mean(scores))
    body.append(row)
    
    fio.WriteMatrix(outputdir + "quality." + str(np) + '.L' + str(L) + ".txt", body, header)

if __name__ == '__main__':
    import sys
    ilpdir = sys.argv[1]
    #ilpdir = "../../data/oracle/"
    
    from config import ConfigFile
    config = ConfigFile()
                    
    for L in [config.get_length_limit()]:
        for np in ['sentence']:
            getQuality(ilpdir, np, L, ilpdir, Lambda = None)
                          
    print "done"