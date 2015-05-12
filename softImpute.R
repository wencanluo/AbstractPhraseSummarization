require('softImpute')

args <- commandArgs(TRUE)
R <- as.integer(args[1])
L <- as.numeric(args[2])
type <- args[3]

setwd("E:/project/AbstractPhraseSummarization/code/AbstractPhraseSummarization")
load("X.gzip")

str(X)
typeof(X)

X[ which(X==0,arr.ind = T) ] = NA
xna=X

fit=softImpute(xna,rank=R,lambda=L,type=type, thresh = 1e-04, maxit = 20, trace.it=TRUE, final.svd=TRUE)
newX = complete(xna,fit)

rank = rankMatrix(newX)[1]
print(paste0("rank:", rank))

filename=paste("newX_",R,"_",L,"_",type,".gzip",sep="")
save(newX, file=filename, compress=TRUE)
