import numpy
from rpy2.robjects import r
import pandas.rpy.common as com
from pandas import DataFrame
import os
import SVD_Test

def TestSoftImpute():
    row, col = 15, 10
    K = 4
    
    ratio = 0.3
    A = numpy.random.choice([0.0, 1.0], size=(row, col), p=[1-ratio, ratio])
    print A
    
    os.system('Rscript softImpute.R')
    
    newA = SoftImpute(A)
    
    SVD_Test.PrintMatrix(newA)
    
def SoftImpute(X):
    df = DataFrame(X)
    df = com.convert_to_r_dataframe(df)
    r.assign("X", df)
    r("save(X, file='X.gzip', compress=TRUE)")
    os.system('Rscript softImpute.R')
    
    return LoadR()

def LoadR():
    r('load("newX.gzip")')
    newX = r['newX']
    newX = numpy.array(newX)
    return newX.T
       
if __name__ == '__main__':
    TestSoftImpute()