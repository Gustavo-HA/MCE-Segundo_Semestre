S <- matrix(c(.35,.15,-.19,
              .15,.13,-.03,
              -.19,-.03,.16), nrow=3)

R <- diag(1/diag(S)^0.5) %*% S %*% diag(1/diag(S)^0.5)

eps <- 0.05

R_inv <- solve(R)

psi_i <- diag(1/(diag(R_inv)))
h_i <- 1 - diag(psi_i)

R_T <- R - diag(diag(R)) + diag(h_i)
eigvals <- eigen(R_T)

L <- cbind(sqrt(eigvals$values[1])*eigvals$vectors[,1], 
           sqrt(eigvals$values[2])*eigvals$vectors[,2])

h_i <- diag(L%*%t(L))
R_T <- R - diag(diag(R)) + diag(h_i)
eigvals <- eigen(R_T)

L <- cbind(sqrt(eigvals$values[1])*eigvals$vectors[,1], 
           sqrt(eigvals$values[2])*eigvals$vectors[,2])


i = 0
for(i in 1:10){
  R_inv <- solve(R)
  
  psi_i <- diag(1/(diag(R_inv)))
  h_i <- 1 - diag(psi_i)
  
  R_T <- R - diag(diag(R)) + diag(h_i)
  eigvals <- eigen(R_T)
  
  L <- cbind(sqrt(eigvals$values[1])*eigvals$vectors[,1], 
             sqrt(eigvals$values[2])*eigvals$vectors[,2])
  
  R <- L%*% t(L)
}
