R <- matrix(c(1.00, 0.02, 0.96, 0.42, 0.01,
              0.02, 1.00, 0.13, 0.71, 0.85,
              0.96, 0.13, 1.00, 0.50, 0.11,
              0.42, 0.71, 0.50, 1.00, 0.79,
              0.01, 0.85, 0.11, 0.79, 1.00),
            nrow = 5, byrow = TRUE)

print(R)

valores_vec_propios <- eigen(R)

# Con criterio de Kaiser elegimos 2 factores
valores_vec_propios$values[1] / 5+valores_vec_propios$values[2] / 5

# Ahora construimos las cargas del modelo con 2 factores

vec_1 <- sqrt(valores_vec_propios$values[1])*valores_vec_propios$vectors[,1]

vec_2 <- sqrt(valores_vec_propios$values[2])*valores_vec_propios$vectors[,2]

# Matriz de cargas
L <- cbind(vec_1,vec_2)

# Comunalidades
comunalidades <- rowSums(L^2)

print(comunalidades)


# Varianza especifica
psi <- diag(diag(R - L %*% t(L)))


psi + L %*% t(L)
