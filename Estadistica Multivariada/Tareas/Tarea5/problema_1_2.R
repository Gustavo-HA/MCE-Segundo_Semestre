########### PROBLEMA 1 ###########

# Inciso (a)
# Matriz de datos de animales 
datos_animales <- matrix(
  c(
    # León
    1, 1, 0, 0, 1, 1,
    # Jirafa
    1, 1, 1, 0, 0, 1,
    # Vaca
    1, 0, 0, 1, 0, 1,
    # Oveja
    0, 0, 0, 1, 0, 1,
    # Gato doméstico
    1, 0, 0, 0, 1, 1,
    # Hombre
    0, 0, 0, 0, 1, 0
  ),
  nrow = 6,           # Número de animales (renglones)
  byrow = TRUE        # Llenar la matriz por renglones
)

# Inciso (b)

# a = Número de variables con respuesta 1 en ambos animales
# b = Número de variables con respuesta 0 en el primer animal y con respuesta 1
# en el segundo animal
# c = numero de variables con respuesta 1 en el primer animal y con respuesta 0
# en el segundo animal
# d = Número de variables con respuesta 0 en ambos individuos
# p = Número total de variables

# sokal-michener_ij = (a+d)/p ; jacard_ij = a/(a+b+c)
# Definición de la función sokal-michener
sokal_michener <- function(x, y) {
  # x = vector de respuestas del primer animal
  # y = vector de respuestas del segundo animal
  a <- sum(x & y)
  b <- sum(!x & y)
  c <- sum(x & !y)
  d <- sum(!x & !y)
  p <- length(x)
  return((a + d) / p)
}

# Definición de la función jacard
jacard <- function(x, y) {
  # x = vector de respuestas del primer animal
  # y = vector de respuestas del segundo animal
  a <- sum(x & y)
  b <- sum(!x & y)
  c <- sum(x & !y)
  return(a / (a + b + c))
}

# Calculando la matriz de similaridades con Sokal-Michener
matriz_sokal_michener <- matrix(0, nrow = 6, ncol = 6)
for (i in 1:6) {
  for (j in 1:6) {
    matriz_sokal_michener[i, j] <- sokal_michener(datos_animales[i, ], datos_animales[j, ])
  }
}

# Calculando la matriz de similaridades con Jacard
matriz_jacard <- matrix(0, nrow = 6, ncol = 6)
for (i in 1:6) {
  for (j in 1:6) {
    matriz_jacard[i, j] <- jacard(datos_animales[i, ], datos_animales[j, ])
  }
}


## Matriz de distancias D^2 = 2(1 - S)

matriz_unos = matrix(1, nrow = 6, ncol = 6)

# D^2 Sokal-Michener:
matriz_distancia_sokal_michener <- 2 * (matriz_unos - matriz_sokal_michener)

# D^2 Jacard:
matriz_distancia_jacard <- 2 * (matriz_unos - matriz_jacard)



########### PROBLEMA 2 ###########

# Inciso (a)

# Obtener B, sus primeros eigenvectores, y obtener las coordenadas principales.
n <- nrow(datos_animales)
H = diag(rep(1,6)) - (1/n) * matrix(1, nrow = n, ncol = n)
B <- -(0.5) * H %*% matriz_distancia_sokal_michener %*% H

# Obtener los eigenvectores y eigenvalores
eigen_B <- eigen(B)
cumsum(eigen_B$values / sum(eigen_B$values)* 100) # 2 eigenvalores suficientes
m = 2

# Obtener matriz de coord princ X_m
X_m <- eigen_B$vectors[, 1:m] %*% diag(sqrt(eigen_B$values[1:m]))


plot(X_m[, 1], X_m[, 2], xlab = "Dim 1", ylab = "Dim 2",
     main = "Coordenadas principales de los animales", xlim = c(-1, 1),
     ylim = c(-1, 1))
text(X_m[, 1], X_m[, 2], labels = c("León", "Jirafa", "Vaca", "Oveja", "Gato", "Hombre"),
     pos = 4, cex = 0.7)


## Inciso (b)

elefante <- c(1, 1, 0, 0, 0, 1)
similaridades_elefante <- rep(0,7)
similaridades_elefante[7] <- 1

for (i in 1:6){
  similaridades_elefante[i] <- sokal_michener(elefante, datos_animales[i,])
}
similaridades_elefante

matriz_sokal_michener <- cbind(rbind(matriz_sokal_michener,0),0)
matriz_sokal_michener[7,] <- similaridades_elefante
matriz_sokal_michener[,7] <- similaridades_elefante

matriz_unos = matrix(1, nrow = 7, ncol = 7)
matriz_distancia_sokal_michener_elef <- 2 * (matriz_unos - matriz_sokal_michener)

d <- matriz_distancia_sokal_michener_elef[7,1:6]

b <- diag(X_m %*% t(X_m))

x <- 0.5 * diag(1/eigen_B$values[1:m]) %*% t(X_m) %*% (b-d)

X_m <- rbind(X_m, t(x))

plot(X_m[, 1], X_m[, 2], xlab = "Dim 1", ylab = "Dim 2",
     main = "Coordenadas principales de los animales", xlim = c(-1, 1),
     ylim = c(-1, 1))
text(X_m[, 1], X_m[, 2], labels = c("León", "Jirafa", "Vaca", 
                                    "Oveja", "Gato", "Hombre", "Elefante"),
     pos = 4, cex = 0.7)



