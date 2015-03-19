require('softImpute')
setwd("E:/project/AbstractPhraseSummarization/code/AbstractPhraseSummarization")
load("X.gzip")
X[ which(X==0,arr.ind = T) ] = NA
xna=X

fit=softImpute(xna,rank=50,lambda=1.5,type="svd")
newX = complete(xna,fit)
save(newX, file='newX.gzip', compress=TRUE)
