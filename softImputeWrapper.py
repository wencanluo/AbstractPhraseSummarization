#http://cran.us.r-project.org/web/packages/softImpute/
import numpy
from rpy2.robjects import r
import pandas.rpy.common as com
from pandas import DataFrame
import os
import fio

def LoadX(path = './'):
    filename = path + 'X.gzip'
    if not fio.IsExist(filename):
        return None
    
    cmd = 'load("'+filename+'")'
    
    r(cmd)
    X = r['X']
    X = numpy.array(X)
    
    return X
    
def LoadR(rank, Lambda, type='svd', path='./'):
    filename = path + 'newX_'+str(rank)+'_'+str(Lambda)+'_'+str(type)+'.gzip'
    if not fio.IsExist(filename):
        return None
    
    cmd = 'load("'+filename+'")'
    
    r(cmd)
    newX = r['newX']
    newX = numpy.array(newX)
    
    return newX.T
    
#matrix competition using the SoftImpute algorithm
#X: input matrix
#R: max rank
#L: lambda
def SoftImpute(X, rank, Lambda, name, folder, type='svd'):
    df = DataFrame(X)
    df = com.convert_to_r_dataframe(df)
    r.assign('X', df)
    r("save(X, file='%s/%s.gzip', compress=TRUE)" % (folder, name)) #(os.path.join(folder, name)))
     
    #cmd = 'Rscript softImpute.R %d %f %s' % (rank, Lambda, type)
    #cmd = 'Rscript softImputeDemo.R %d %f %s' % (rank, Lambda, type)
    #print cmd
    #os.system(cmd)
    
    #return LoadR(rank, Lambda, type)

def LoadA(name, folder):
    filename = os.path.join(folder, name +'.gzip')
    filename = filename.replace('\\', '/')
    if not fio.IsExist(filename):
        return None
    
    cmd = 'load("'+filename+'")'
    
    r(cmd)
    newX = r['X']
    newX = numpy.array(newX)
    
    return newX

def LoadMC(Lambda, name, folder, type='svd'):
    fid = name +'_'+str(Lambda)+'_'+str(type)
    filename = os.path.join(folder, fid +'.gzip')
    filename = filename.replace('\\', '/')
    print filename
    if not fio.IsExist(filename):
        return None
    
    cmd = 'load("'+filename+'")'
    
    #cmd = 'load("../../data/TAC_MC/s08/D0801A-A_2_svd.gzip")'
    r(cmd)
    newX = r['newX']
    newX = numpy.array(newX)
    
    return newX.T
    #os.system(cmd)
    
    #return LoadR(rank, Lambda, type)
    
def TestSoftImpute():
    row, col = 15, 10
    K = 4
    
    ratio = 0.3
    A = numpy.random.choice([0.0, 1.0], size=(row, col), p=[1-ratio, ratio])
    print A.shape
    print A
    
    newA = SoftImpute(A, K, 1.5)
    
    print newA.shape
    
    import SVD_Test
    SVD_Test.PrintMatrix(newA)
           
if __name__ == '__main__':
    TestSoftImpute()