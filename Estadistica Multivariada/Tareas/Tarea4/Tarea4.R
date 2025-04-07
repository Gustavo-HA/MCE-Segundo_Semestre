library("MASS")
library("stats")
library("scatterplot3d")
library("psych") #prueba de esfericidad de bartlett

### Problema 1

mu <- matrix(c(0,1,1), nrow=3)
sigma <- matrix(c(3,-4,2,
                  -4,12,-2,
                  2,-2,3), nrow=3)
R <- diag(diag(1/sigma)**0.5) %*% sigma %*% diag(diag(1/sigma)**0.5) 

pca <- princomp(R)
screeplot(pca)

L <- matrix(pca$loadings[,1], nrow=3)
comun <- diag((L%*%t(L)))

varesp <- diag(1-comun)

varesp

L %*% t(L) + varesp
