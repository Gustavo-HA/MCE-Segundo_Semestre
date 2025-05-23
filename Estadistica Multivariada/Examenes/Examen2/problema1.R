
X = matrix(c(1,0.63,0.45,
             0.63,1,0.35,
             0.45,0.35,1), nrow=3, byrow=TRUE)

library("psych")

l_2 <- (0.35*0.63/0.45)^(0.5)
l_3 <- (0.45/0.63)*l_2
l_1 <- 0.63/l_2
l_1^2
l_2^2
l_3^2

1-l_1^2
1-l_2^2
1-l_3^2

n <- 24
S <- 10^(-3)*matrix(c(11.072,8.019,8.160,
                       8.019,6.417,6.005,
                       8.160,6.005,6.773), nrow=3, byrow=TRUE)

S_n <- S * (n-1)/n

L <- matrix(c(0.1022,
              0.0752,
              0.0765), nrow=3, byrow=TRUE)

Psi <- S - L %*% t(L)

S_n - L %*% t(L) - Psi

p <- 3
m <- 1

det_sup <- det(L %*% t(L) + Psi)
det_inf <- det(S_n)

(n - 1 - (2*p + 4*m + 5/6))*log(det_sup/det_inf)

((p-m)^2 - p - m)/2
