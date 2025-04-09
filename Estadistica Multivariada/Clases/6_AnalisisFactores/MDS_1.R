
Delta <- matrix(c(
  0,   627,  351,  550,  488,  603,
  627,   0,  361, 1043,  565, 1113,
  351,  361,    0,  567,  564,  954,
  550, 1043,  567,    0,  971,  950,
  488,  565,  564,  971,    0,  713,
  603, 1113,  954,  950,  713,    0),nrow = 6, ncol = 6, byrow=TRUE)

n <- 6

I <- diag(n)

ones <- rep(1, n)

# Construye la matriz de centrado
H <- I - (1 / n) * (ones %*% t(ones))

A <- -1/2 * Delta^2

B <- H %*% A %*% H

B <- 1/100000 * B
Resultados_B <- eigen(B)

print("Valores propios")
print(Resultados_B$values)

print(" Vectores propios")
print(Resultados_B$vectors)


print(" Porcentaje de varianza total")
print("Para el valor propio 1:")
print(Resultados_B$values[1]/sum(abs(Resultados_B$values)))
print("Hasta el segundo valor propio")
print(sum(Resultados_B$values[1:2])/sum(abs(Resultados_B$values)))



lambda <- Resultados_B$values
Vectors <- Resultados_B$vectors

# Seleccionamos las dos primeras componentes principales (las que más varianza explican)
# Solo usamos los valores propios positivos (por si las moscas hay valores negativos pequeños)
X <- Vectors[, 1:2] %*% diag(sqrt(pmax(lambda[1:2], 0)))
# Etiquetas de las ciudades
cities <- c("Madrid", "Barcelona", "Valencia", "Sevilla", "San Sebastián", "La Coruña")

# Plot en 2D
plot(X, type = "n", main = "Escalamiento Multidimensional Clásico (MDS)", xlab = "Dimensión 1", ylab = "Dimensión 2")
text(X, labels = cities, col = "blue", cex = 1.2)





