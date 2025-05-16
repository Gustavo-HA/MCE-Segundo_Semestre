# Cargar los datos completos de los 50 jugadores
data <- data.frame(
  Nombre = c("Ronaldinho", "Etoo", "Xavi", "Messi", "Puyol", "Raul", "Ronaldo", "Beckham", "Casillas", "Cannavaro",
              "Torres", "Aguero", "Maxi", "Pablo", "Maniche", "Morientes", "Joaquin", "Villa", "Ayala", "Canizares",
              "Jesus Navas", "Puerta", "Javi Navarro", "Daniel Alves", "Kanouté", "Valerón", "Amendini", "Capdevila", "Riki", "Coloccini",
              "Riquelme", "Forlan", "Cani", "Javi Venta", "Tachinardi", "Pandiani", "Tamudo", "De la Peña", "Luis Garcia", "Jonathan",
              "Aimar", "Diego Milito", "Savio", "Sergio Garcia", "Zapater", "Edu", "Juanito", "Melli", "Capi", "Doblas"),
  
  X1 = c(15,21,6,7,1,7,18,4,0,0,
         24,14,10,3,3,13,5,22,1,0,
         2,6,7,2,12,9,8,3,7,2,
         10,17,4,0,4,6,10,2,8,4,
         6,9,3,7,5,6,2,5,7,0),
  
  X2 = c(26,25,26,19,28,29,30,31,25,33,
         22,18,25,25,29,30,25,24,33,36,
         20,21,32,23,29,31,22,28,26,24,
         28,27,25,30,31,30,28,30,25,21,
         26,27,32,23,21,27,30,22,29,25),
  
  X3 = c(1.78,1.8,1.7,1.69,1.78,1.8,1.83,1.8,1.85,1.76,
         1.83,1.72,1.8,1.92,1.73,1.86,1.79,1.75,1.77,1.81,
         1.70,1.83,1.82,1.71,1.92,1.84,1.92,1.81,1.86,1.82,
         1.82,1.72,1.75,1.8,1.87,1.84,1.77,1.69,1.8,1.8,
         1.68,1.81,1.71,1.76,1.73,1.82,1.83,1.81,1.75,1.84),
  
  X4 = c(71,75,68,67,78,73.5,82,67,70,75.5,
         70,68,79,80,69,79,75,69,75.5,78,
         60,74,75,64,82,71,78,79,80,78,
         75,73,69.5,73,80,74,74,69,68,72,
         60,78,68,69,70.5,74,80,78,73,78),
  
  X5 = c(1,0,0,0,0,1,0,0,0,0,
         0,0,0,0,0,0,0,0,0,1,
         0,1,0,0,1,0,0,1,0,1,
         0,0,0,1,1,0,0,0,0,1,
         1,0,1,0,0,1,0,0,0,0),
  
  X6 = c(2,3,5,1,5,5,2,9,5,4,
         5,1,1,5,8,5,5,5,1,5,
         5,5,5,2,6,5,5,5,5,1,
         1,7,5,5,4,7,5,5,5,5,
         1,1,2,5,5,2,5,5,5,5),
  
  X7 = c(2,2,4,3,3,3,1,3,4,2,
         4,3,3,4,2,3,4,3,1,3,
         3,3,3,2,1,3,3,4,3,2,
         2,3,3,3,4,1,3,3,3,3,
         2,2,2,3,3,3,4,3,2,3)
)

data


G_h <- sapply(data[, 2:5], function(x) max(x) - min(x))
cat("Rango para cada variable cuantitativas continuas:\n")
print(G_h)


gower_similarity <- function(i, j) {
  
  idx_quant <- 2:5   # (goles, edad, altura, peso)
  idx_bin   <- 6     # (pierna buena: 0/1)
  idx_cat   <- 7:8   # (nacionalidad, estudios)

  ## Parte cuantitativa Σ_{h∈p1} (1 - |x_ih - x_jh| / G_h)
  cuantitativas <- sum(
    1 - abs(data[i, idx_quant] - data[j, idx_quant]) / G_h
  )

  ## Parte binaria  (a, d)
  a <- sum(data[i, idx_bin] == 1 & data[j, idx_bin] == 1)
  d <- sum(data[i, idx_bin] == 0 & data[j, idx_bin] == 0)

  ## Parte cualitativa no binaria  α
  alfa <- sum(data[i, idx_cat] == data[j, idx_cat])

  ## p_1, p_2, p_3
  p1 <- length(idx_quant)   
  p2 <- length(idx_bin)     
  p3 <- length(idx_cat)     

  ## s_ij Coeficiente de similaridad de Gower
  s_ij <- (cuantitativas + a + alfa) / (p1 + (p2 - d) + p3)

  return(s_ij)
}


matriz_similaridades = matrix(1, nrow = nrow(data), ncol = nrow(data))
for (i in 1:(nrow(data) - 1)) {
  for (j in (i + 1):nrow(data)) {
    matriz_similaridades[i, j] <- gower_similarity(i, j)
    matriz_similaridades[j, i] <- matriz_similaridades[i, j]
  }
}

# Inciso (A): Matriz de distancias de Gower
d_cuadrado <- 1 - matriz_similaridades
D <- sqrt(d_cuadrado)
rownames(D) <- colnames(D) <- data$Nombre


# Inciso (B): Obtener coordenadas principales, % de variabilidad
# y semejanza entre jugadores.

n <- nrow(d_cuadrado)
H <- diag(n) - (1 / n) * matrix(1, n, n)
B <- -(0.5) * H %*% d_cuadrado %*% H

eigen_B <- eigen(B)
eigen_B$values <- eigen_B$values[eigen_B$values > 0] 
eigen_B$vectors <- eigen_B$vectors[, eigen_B$values > 0]

cumsum(eigen_B$values) / sum(eigen_B$values) # 33% con 2 coord princ
m <- 2

# Obtenemos las coordenadas principales
X_m <- eigen_B$vectors[, 1:m] %*% diag(sqrt(eigen_B$values[1:m]))
rownames(X_m) <- data$Nombre


factor <- 0.4
plot(X_m[, 1], X_m[, 2], xlab = "Dim 1", ylab = "Dim 2",
     main = "Coordenadas principales de los jugadores", xlim = factor*c(-1, 1),
     ylim = factor*c(-1, 1))
text(X_m[, 1], X_m[, 2], labels = rownames(X_m),
     pos = 4, cex = 0.7)







