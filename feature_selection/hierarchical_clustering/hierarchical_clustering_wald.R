#!/usr/bin/ Rscript
args <- commandArgs(trailingOnly=TRUE)

path <- paste(args[1], "output/nonescludere.txt", sep = "")
print(path)

lista <- read.csv(path, sep = '\t', header = 0)
lista$V1 <- gsub('feat', '', lista$V1)
lista <- as.matrix(transform(lista, V1 = as.numeric(V1)))

path = paste(args[1], "output/spear_corr.txt", sep = "")

corr <- read.csv(path, sep = '\t', header = 0)
corr <- corr[lista, lista]
corr <- as.matrix(corr)

abs.corr <- abs(corr)
abs.corr[!is.finite(abs.corr)] <- 0
hc=hclust(as.dist(1-abs.corr),method = "ward.D", members = NULL)

trials = c(11,22,44,66,88,110,165) 

p_path = paste(args[1], "output/g_", sep = "")

for(i in 1:length(trials)) 
{
  #print(i)
  path <- paste(p_path, toString(trials[i]), ".txt", sep = "")
  g <- cutree(hc, k = trials[i])
  write.csv(g, file = path, sep = "\t", col.names = FALSE, quote = FALSE)
}
