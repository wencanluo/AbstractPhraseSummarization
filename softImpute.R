require('softImpute')

args <- commandArgs(TRUE)
R <- as.integer(args[1])
L <- as.numeric(args[2])
type <- args[3]

setwd("E:/project/AbstractPhraseSummarization/code/AbstractPhraseSummarization")
load("X.gzip")

X[ which(X==0,arr.ind = T) ] = NA
xna=X

fit=softImpute(xna,rank=R,lambda=L,type=type, thresh = 1e-04, maxit = 50, trace.it=TRUE, final.svd=FALSE)
newX = complete(xna,fit)
filename=paste("newX_",R,"_",L,"_",type,".gzip",sep="")
save(newX, file=filename, compress=TRUE)
