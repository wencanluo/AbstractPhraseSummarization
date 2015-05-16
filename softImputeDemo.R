#!/usr/bin/Rscript

require('softImpute')


#rankMaxs <- c(10, 50, 100, 500, 1000)
#rankMaxs <- 2690
rankMaxs <- 200
#lambdas <- 10^seq(from = -5, to = 1)
#lambdas <- c(10, 5, 1, 0.1)
lambdas <- c(2, 1.5, 1, 0.5, 0.1)

ranks <- rep(1, times = length(lambdas))

setwd("E:/project/AbstractPhraseSummarization/data/matrix/exp1/")
load("X.gzip")

X = as.matrix(X)
print(dim(X))

X[ which(X==0, arr.ind = T) ] = NA
xna=X

for (j in 1:length(rankMaxs)){
	rankMax = rankMaxs[j]
	
	warm = NULL
	
	for( i in 1:length(lambdas)) {		
		fit=softImpute(xna, rank=rankMax, lambda = lambdas[i], thresh = 1e-04, maxit = 100, trace.it=TRUE, warm=warm)
		newX = complete(xna,fit)
		
		ranks[i]=sum(round(fit$d, 3)>0)
		warm=fit
		
		cat(i,"lambda=",lambdas[i],"rank.max",rankMax,"rank",ranks[i],"\n")
		
		filename=paste("newX_", rankMax, "_", lambdas[i], "_svd.gzip", sep="")
	}
}

