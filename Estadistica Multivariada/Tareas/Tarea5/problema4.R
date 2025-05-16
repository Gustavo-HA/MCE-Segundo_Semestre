rm(list = ls())

library(smacof)

# Directorios
ruta_datos <- "C:/Users/uzgre/Documentos/R codes/Tarea5_Estadistica_Multivariada"
ruta_grafico <- "C:/Users/uzgre/Documentos/R codes"

# Establecer el directorio de trabajo
setwd(ruta_datos)

# ===============================
# DATOS: PROPORCIONES GÉNICAS
# ===============================
# Tabla con proporciones observadas de grupos sanguíneos por población
tabla_sanguinea <- data.frame(
  Grupo_A  = c(0.21, 0.25, 0.22, 0.19, 0.18, 0.23, 0.30, 0.10, 0.27, 0.21),
  Grupo_AB = c(0.06, 0.04, 0.06, 0.04, 0.00, 0.00, 0.00, 0.06, 0.04, 0.05),
  Grupo_B  = c(0.06, 0.14, 0.08, 0.02, 0.15, 0.28, 0.06, 0.13, 0.06, 0.20),
  Grupo_O  = c(0.67, 0.57, 0.64, 0.75, 0.67, 0.49, 0.64, 0.71, 0.63, 0.54)
)
poblaciones <- c("Francesa", "Checa", "Germanica", "Vasca", "China",
                 "Ainu", "Esquimal", "Afroamericana USA", "Española", "Egipcia")

print(tabla_sanguinea)

# ===============================
# INCISO(A): CÁLCULO DE DISTANCIAS BHATTACHARYYA
# ===============================
prop_gen <- as.matrix(tabla_sanguinea)
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
print(round(d2ij, 4))

# ===============================
# INCISO(B): ANÁLISIS MDS CLÁSICO, COORDENADAS PRINCIPALES
# ===============================

# Construcción de la matriz B con doble centrado
I <- diag(n)
uno <- rep(1, n)
H <- I - (1/n) * (uno %*% t(uno))

A <- -0.5 * d2ij^2
B <- H %*% A %*% H

# Descomposición espectral de B
Resultados_B <- eigen(B)
valores_propios <- Resultados_B$values
vectores_propios <- Resultados_B$vectors

cat("Valores propios:\n")
print(round(valores_propios, 4))

cat("\nVectores propios:\n")
print(round(vectores_propios, 4))


# ===============================
# INCISO(C): DIMENSION ADECUADA 
# ===============================

# Porcentaje de varianza explicada
var_total <- sum(abs(valores_propios))
varianza_1 <- valores_propios[1] / var_total
varianza_12 <- sum(valores_propios[1:2]) / var_total

cat("\nPorcentaje de varianza explicada:\n")
cat(sprintf(" - Primer valor propio: %.4f\n", varianza_1))
cat(sprintf(" - Dos primeros valores propios: %.4f\n", varianza_12))

# ===============================
# COORDENADAS PRINCIPALES Y GRÁFICA
# ===============================
# Solo se consideran componentes asociados a autovalores positivos
X <- vectores_propios[, 1:2] %*% diag(sqrt(valores_propios[1:2]))

# Gráfico de la configuración en 2D
plot(X, type = "n", xlab = "Dim 1", ylab = "Dim 2", main = "MDS clásico")
text(X, labels = poblaciones, col = "blue", cex = 1.0)


# ===============================
# INCISO(D): ANÁLISIS MDS MINIMOS CUADRADOS
# ===============================

# Convertimos la matriz de distancias 
d_bhat <- as.dist(d2ij)

# Transformacion tipo razon
mds_ratio <- mds(d_bhat, type = "ratio", init = X, ndim = 2)
plot(mds_ratio$conf, type = "n", main = "MDS tipo razón", xlab = "Dim 1", ylab = "Dim 2")
text(mds_ratio$conf, labels = poblaciones, col = "darkgreen", cex = 1.0)

# Transformacion tipo intervalo
mds_interval <- mds(d_bhat, type = "interval", init = X, ndim = 2)
plot(mds_interval$conf, type = "n", main = "MDS tipo intervalo", xlab = "Dim 1", ylab = "Dim 2")
text(mds_interval$conf, labels = poblaciones, col = "darkred", cex = 1.0)

# Transformacion tipo ordinal 
mds_ordinal <- mds(d_bhat, type = "ordinal", init = X, ndim = 2)
plot(mds_ordinal$conf, type = "n", main = "MDS tipo ordinal", xlab = "Dim 1", ylab = "Dim 2")
text(mds_ordinal$conf, labels = poblaciones, col = "blue", cex = 1.0)


# Comparacion de los tres modelos y dimensionalidad

cat("Stress (ratio):", mds_ratio$stress, "\n")
cat("Stress (interval):", mds_interval$stress, "\n")
cat("Stress (ordinal):", mds_ordinal$stress, "\n")

# Justificación de la dimensión: si el stress es menor a 0.1, 
# la representación en 2D es considerada buena. 
# Si está entre 0.1 y 0.2, es razonable. Mayor a 0.2, ya preocupa.





