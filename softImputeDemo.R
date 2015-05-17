#!/usr/bin/Rscript

require('softImpute')

rankMaxs <- c(2690)
lambdas <- c(1, 0.1)

ranks <- rep(1, times = length(lambdas))

exps <- c(1)

for (k in 1:length(exps)){
	id = exps[k]
	folder = paste("E:/project/AbstractPhraseSummarization/data/matrix/exp", id, sep="")
	cat("folder=", folder)

	setwd(folder)
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
			fit_filename=paste("fit_", rankMax, "_", lambdas[i], "_svd.gzip", sep="")
			save(fit, file=fit_filename, compress=TRUE)
		
			newX = complete(xna,fit)
		
			ranks[i]=sum(round(fit$d, 3)>0)
			warm=fit
		
			cat(i,"lambda=",lambdas[i],"rank.max",rankMax,"rank",ranks[i],"\n")
		
			filename=paste("newX_", rankMax, "_", lambdas[i], "_svd.gzip", sep="")
			save(newX, file=filename, compress=TRUE)
		}
	}
}
