library(smacof)

tabla <- data.frame(
  Grupo_A  = c(0.21, 0.25, 0.22, 0.19, 0.18, 0.23, 0.30, 0.10, 0.27, 0.21),
  Grupo_AB = c(0.06, 0.04, 0.06, 0.04, 0.00, 0.00, 0.00, 0.06, 0.04, 0.05),
  Grupo_B  = c(0.06, 0.14, 0.08, 0.02, 0.15, 0.28, 0.06, 0.13, 0.06, 0.20),
  Grupo_O  = c(0.67, 0.57, 0.64, 0.75, 0.67, 0.49, 0.64, 0.71, 0.63, 0.54)
)
poblaciones <- c("Francesa", "Checa", "Germanica", "Vasca", "China",
                 "Ainu", "Esquimal", "Afroamericana USA", "Española", "Egipcia")

print(tabla)

# Inciso (a)

prop_gen <- as.matrix(tabla)
n <- nrow(prop_gen)
d2ij <- matrix(0, nrow = n, ncol = n)

for (i in 1:(n - 1)) {
  for (j in (i + 1):n) {
    suma_sqrt <- sum(sqrt(prop_gen[i, ] * prop_gen[j, ]))
    dist <- acos(suma_sqrt)
    d2ij[i, j] <- dist
    d2ij[j, i] <- dist
  }
}

print("Matriz de distancias Bhattacharyya:")
round(d2ij, 3)

# Inciso (B)


H <- diag(n) - (1/n) * rep(1,n) %*% t(rep(1,n))
B <- -0.5 * H %*% d2ij %*% H

# eig B
eigen_B <- eigen(B)
eigen_B$values <- eigen_B$values[eigen_B$values > 0]
eigen_B$vectors <- eigen_B$vectors[, eigen_B$values > 0]


# Inciso (C)

# Porcentaje de varianza explicada
var.acum <- cumsum(eigen_B$values) / sum(eigen_B$values)
plot(var.acum, type = "b", xlab = "N Vectores Propios", ylab = "Varianza acumulada",
     main = "Porcentaje de varianza explicada")

plot(eigen_B$values / sum(eigen_B$values), type = "b", xlab = "N Vectores Propios", ylab = "Contribución a la varianza",
     main = "Scree-Plot")

m <- 2

X <- eigen_B$vectors[, 1:m] %*% diag(sqrt(eigen_B$values[1:m]))

# Gráfico de la configuración en 2D
plot(1.1*X, type = "n", xlab = "Dim 1", ylab = "Dim 2", main = "MDS a 2 Coordenadas")
text(X, labels = poblaciones, col = "black", cex = 1)


# Inciso (D)

# Convertimos la matriz de distancias 
d_bhat <- as.dist(d2ij)

# Transformacion tipo razon
mds_ratio <- mds(d_bhat, type = "ratio", init = X, ndim = 2)
plot(mds_ratio$conf, type = "n", main = "Ratio", xlab = "Dim 1", ylab = "Dim 2")
text(mds_ratio$conf, labels = poblaciones, col = "darkred", cex = 1.0)

# Transformacion tipo intervalo
mds_interval <- mds(d_bhat, type = "interval", init = X, ndim = 2)
plot(mds_interval$conf, type = "n", main = "Interval", xlab = "Dim 1", ylab = "Dim 2")
text(mds_interval$conf, labels = poblaciones, col = "purple", cex = 1.0)

# Transformacion tipo ordinal 
mds_ordinal <- mds(d_bhat, type = "ordinal", init = X, ndim = 2)
plot(mds_ordinal$conf, type = "n", main = "Ordinal", xlab = "Dim 1", ylab = "Dim 2")
text(mds_ordinal$conf, labels = poblaciones, col = "darkgreen", cex = 1.0)

mds_ratio$stress
mds_interval$stress
mds_ordinal$stress





