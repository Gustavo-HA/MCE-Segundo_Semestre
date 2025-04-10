library("MASS")
library("stats")
library("scatterplot3d")
library("psych") #prueba de esfericidad de bartlett

### Problema 2

R <- matrix(c(
  1, 0.69, 0.28, 0.35,
  0.69, 1, 0.255, 0.195,
  0.28, 0.255, 1, 0.61,
  0.35, 0.195, 0.61, 1
), nrow = 4, byrow = TRUE)

#### Inciso (a)

eig <- eigen(R)
lambdas <- eig$values
v_propios <- eig$vectors

#### Inciso (c)

pca_factorial <- function(R,lambdas,v_propios,m, type = "corr"){
  lambda <- lambdas[1:m]
  vec <- v_propios[,1:m]

  if (type == "corr"){
    n <- nrow(R)  
  }
  else{
    n <- sum(diag(R))
  }
  
  for (i in 1:m){
    vec[,i] <- sqrt(lambda[i]) * vec[,i]
  }
  
  LL <- vec %*% t(vec)
  psi <- diag(R - LL)
  
  comunalidades <- rowSums(vec^2)
  
  varianza_explicada <- (vec^2)*100
  
  var_per_factor <- lambda / n * 100
  
  var_total_acumulado <- cumsum(var_per_factor)
  
  output <- list(
    L = vec,
    psi = diag(psi),
    comm = comunalidades,
    var_ex = varianza_explicada,
    var_factor = var_per_factor,
    var_factor_acum = var_total_acumulado
  )
  return(output)
}

model <- pca_factorial(R, lambdas, v_propios, 2)

model$L
model$psi


#### Inciso (d)
plot(model$var_factor, type = "b",
     main = "Scree Plot",
     xlab = "Factor",
     ylab = "Varianza Explicada (%)",
     pch = 19)

#### Inciso (e)
correlaciones_Z2 <- model$L[2, ]
print(correlaciones_Z2)







## Problema 3

S <- matrix(c(
  0.35, 0.15, -0.19,
  0.15, 0.13, -0.03,
  -0.19, -0.03, 0.16
), nrow = 3, byrow = TRUE)


factor_principal <- function(matriz, m, epsilon, type = "corr"){
  matriz_inv <- solve(matriz)
  psi_anterior <- diag(1/diag(matriz_inv))
  matriz_r <- matriz - psi_anterior
  
  tol <- 1
  contador <- 0
  
  while (tol > epsilon & contador < 100){
    contador <- contador + 1
    cat(sprintf("---%s iteración---\n", contador))
    
    matriz_r <- (matriz_r + t(matriz_r)) / 2 # Forzar simetría
    eig_valores <- eigen(matriz_r)$values
    eig_valores[eig_valores<0] <- 0
    
    eig_vectores <- eigen(matriz_r)$vectors
    
    
    fact <- pca_factorial(matriz_r, eig_valores, eig_vectores, m, type)
    
    psi_nuevo <- diag(matriz - fact$comm)
    # Actualizar matriz_r
    matriz_r <- matriz - psi_nuevo

    L <- fact$L
    tol <- max(abs(psi_nuevo-psi_anterior))
    cat(sprintf("Tolerancia: %f\n", tol))
    psi_anterior <- psi_nuevo
  }
  
  return(list(L = L, psi = psi_nuevo, matriz_r = matriz_r))
}

resultado_S <- factor_principal(S, 2, 0.05)
resultado_S$L %*% t(resultado_S$L) + resultado_S$psi
S
resultado_R <- factor_principal(cov2cor(S), 2, 0.05)
resultado_R$L %*% t(resultado_R$L) + resultado_R$psi
cov2cor(S)

resultado_S$L
resultado_R$L

resultado_S$psi
resultado_R$psi


pca_factorial(S, eigen(S)$values, eigen(S)$vectors, 2)$L
pca_factorial(cov2cor(S), eigen(cov2cor(S))$values, eigen(cov2cor(S))$vectors, 2)$L




## Problema 4

library(MASS)

mu = rep(0,4)
L = matrix(c(0.9, 0.05,
             0.8, 0.3,
             0.2, 0.95,
             0.3, 0.9,
             0.7, 0.15), nrow=5, byrow=TRUE)
psi = diag(c(0.2,0.3,0.1,0.2,0.3))

p <- nrow(L)
m <- ncol(L)

n <- 1000

set.seed(42)

# Muestra F
muestra_F <- matrix(rnorm(n*m), nrow = m, ncol = n)

# Muestra epsilon
mean_epsilon = rep(0,p)
muestra_epsilon = mvrnorm(n = n, mu = mean_epsilon, Sigma = psi)
muestra_epsilon = t(muestra_epsilon)

# Muestra de X
muestra_X <- t(L %*% muestra_F + muestra_epsilon)


# Obtener cargas y Psi de la matriz de datos generados
factanalysis <- factanal(muestra_X, factors = 2)


muestra_X <- t(factanalysis$loadings %*% muestra_F + muestra_epsilon)
factanalysis <- factanal(muestra_X, factors = 2)
factanalysis$loadings




## Problema 5
vendedores <- read.csv2("datosvendedores.csv", sep = ",")
colnames(vendedores) <- c("vendedor","x1","x2", "x3","x4", "x5","x6","x7")

summary(vendedores)
columnas_a_convertir <- 2:4
vendedores[, columnas_a_convertir] <- lapply(vendedores[, columnas_a_convertir], as.numeric)

X <- vendedores[2:8]

#### Inciso (a)
X <- scale(X)


fa2 <- factanal(X, factors = 2)
fa3 <- factanal(X, factors = 3)
fa2$loadings
fa3$loadings

# para m=2

comunalidades_m2 <- rowSums(fa2$loadings^2)
comunalidades_m2
especificas_m2 <- 1 - comunalidades_m2
diag(especificas_m2)
reconstruccion_m2 <- fa2$loadings %*% t(fa2$loadings) + diag(especificas_m2)
reconstruccion_m2

norm(cor(X) - reconstruccion_m2, type = "F")


# para m=3

comunalidades_m3 <- rowSums(fa3$loadings^2)
comunalidades_m3
especificas_m3 <- 1 - comunalidades_m3
especificas_m3
reconstruccion_m3 <- fa3$loadings %*% t(fa3$loadings) + diag(especificas_m3)
reconstruccion_m3

norm(cor(X) - reconstruccion_m3, type = "F")


scores_pond <- factanal(X, factors = 3, scores = "Bartlett")$scores
scores_reg <- factanal(X, factors = 3, scores = "regression")$scores
dimnames(scores_reg)[[1]]<-vendedores[,1]
dimnames(scores_pond)[[1]]<-vendedores[,1]

# scatter en 3d
library(scatterplot3d)
scatterplot3d(scores_reg, angle=35, col.grid="lightblue", main="Grafica de los factor scores", pch=20)
scatterplot3d(scores_pond, angle=35, col.grid="lightblue", main="Grafica de los factor scores", pch=20)
#vamos a graficar los factor scores tomados dos a dos


#f1 x f2
par(pty="s", mar = c(4,4,1,1))
plot(scores_reg[,1],scores_reg[,2],
     ylim=range(scores_reg[,1]),
     xlab="Factor 1",ylab="Factor 2",type="n",lwd=2)
text(scores_reg[,1],scores_reg[,2],
     labels=abbreviate(row.names(scores_reg),minlength=8),cex=0.6,lwd=2)
text(scores_pond[,1],scores_pond[,2],
     labels=abbreviate(row.names(scores_pond),minlength=8),cex=0.6,lwd=2)


#f1 x f3
par(pty="s", mar = c(4,4,1,1))
plot(scores_reg[,1],scores_reg[,3],
     ylim=range(scores_reg[,1]),
     xlab="Factor 1",ylab="Factor 3",type="n",lwd=2)
text(scores_reg[,1],scores_reg[,3],
     labels=abbreviate(row.names(scores_reg),minlength=8),cex=0.6,lwd=2)
text(scores_pond[,1],scores_pond[,3],
     labels=abbreviate(row.names(scores_pond),minlength=8),cex=0.6,lwd=2)


#f2 x f3
par(pty="s", mar = c(4,4,1,1))
plot(scores_reg[,2],scores_reg[,3],
     ylim=range(scores_reg[,2]),
     xlab="Factor 2",ylab="Factor 3",type="n",lwd=2)
text(scores_reg[,2],scores_reg[,3],
     labels=abbreviate(row.names(scores_reg),minlength=8),cex=0.6,lwd=2)
text(scores_pond[,2],scores_pond[,3],
     labels=abbreviate(row.names(scores_pond),minlength=8),cex=0.6,lwd=2)

