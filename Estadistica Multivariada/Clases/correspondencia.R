tabla_contingencia <- matrix(c(688, 116, 584, 188, 4,
                               326, 38, 241, 110, 3,
                               343, 84, 909, 412, 26,
                               98, 48, 403, 681, 85), nrow = 4, byrow = TRUE)

n <- nrow(tabla_contingencia)*ncol(tabla_contingencia)
F <- tabla_contingencia / n

r = F %*% matrix(1, nrow = ncol(F), ncol = 1)
D_r <- diag(as.list(r))
D_r_inv = diag(as.list(1/D_r))
D_r_sqrt <- diag(as.list(sqrt(r)))

c = t(F) %*% matrix(1, nrow = nrow(F), ncol = 1)
D_c_inv_sqr <- diag(as.list(1/sqrt(c)))

R <- (D_r_inv) %*% F

Y <- R %*% D_c_inv_sqr

Z <- D_r_sqrt %*% Y

ZZ <- t(Z) %*% Z
# ZZ <- t(Y) %*% D_r %*% Y ; es lo mismo que ^
