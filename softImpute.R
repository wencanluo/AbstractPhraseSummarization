require('softImpute')

args <- commandArgs(TRUE)
R <- as.integer(args[1])
L <- as.numeric(args[2])
type <- args[3]

setwd("E:/project/AbstractPhraseSummarization/code/AbstractPhraseSummarization")
load("X.gzip")

X[ which(X==0,arr.ind = T) ] = NA
xna=X

fit=softImpute(xna,rank=R,lambda=L,type=type, trace.it=TRUE)
newX = complete(xna,fit)
save(newX, file='newX.gzip', compress=TRUE)
