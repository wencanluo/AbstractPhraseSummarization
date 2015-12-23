import numpy as np
import scipy.stats

def ttest(X, Y, tail=1, type=1):
    '''
    tail: 1 or 2
        tail = 1, one-tail
        tail = 2, two-tail
    type:
        1, paired
        2, unpaired
    '''

    if type == 1:
        t, p = scipy.stats.ttest_rel(X, Y)
    else:
        t, p = scipy.stats.ttest_ind(X, Y)
            
    if tail == 1:
        return t, p/2
    else:
        return t, p

if __name__ == '__main__':
    np.random.seed(12345678)
    rvs1 = scipy.stats.norm.rvs(loc=5,scale=10,size=500)
    print rvs1
    
    rvs2 = scipy.stats.norm.rvs(loc=5,scale=10,size=500)
    print scipy.stats.ttest_rel(rvs1,rvs2)