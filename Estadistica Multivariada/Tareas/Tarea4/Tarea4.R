library("MASS")
library("stats")
library("scatterplot3d")
library("psych") #prueba de esfericidad de bartlett

### Problema 1

mu <- matrix(c(0,1,1), nrow=3)
sigma <- matrix(c(3,-4,2,
                  -4,12,-2,
                  2,-2,3), nrow=3)

pca <- prcomp(sigma)
