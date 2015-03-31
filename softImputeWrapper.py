#http://cran.us.r-project.org/web/packages/softImpute/
import numpy
from rpy2.robjects import r
import pandas.rpy.common as com
from pandas import DataFrame
import os
import SVD_Test

def LoadR():
    r('load("newX.gzip")')
    newX = r['newX']
    newX = numpy.array(newX)
    return newX.T
    
#matrix competition using the SoftImpute algorithm
#X: input matrix
#R: max rank
#L: lambda
def SoftImpute(X, rank, Lambda, type='svd'):
    df = DataFrame(X)
    df = com.convert_to_r_dataframe(df)
    r.assign("X", df)
    r("save(X, file='X.gzip', compress=TRUE)")
     
    cmd = 'Rscript softImpute.R %d %.1f %s' % (rank, Lambda, type)
    print cmd
    os.system(cmd)
    
    return LoadR()

def TestSoftImpute():
    row, col = 15, 10
    K = 4
    
    ratio = 0.3
    A = numpy.random.choice([0.0, 1.0], size=(row, col), p=[1-ratio, ratio])
    print A.shape
    print A
    
    newA = SoftImpute(A, K, 1.5)
    
    print newA.shape
    SVD_Test.PrintMatrix(newA)
           
if __name__ == '__main__':
    TestSoftImpute()