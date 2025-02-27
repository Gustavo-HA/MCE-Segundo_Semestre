
x = c(2,8,6,8)
y = c(12,9,9,10)

df <- data.frame(x=x, y=y)

library(ellipse)

cov_matrix <- cov(df)

media <- colMeans(df)

d_elipse =ellipse(cov_matrix, centre = media, level = 0.5) 

library(ggplot2)

ggplot() +
  geom_point(data=df, aes(x=x,y=y))+
  geom_path(data=d_elipse, aes(x=x,y=y), color = "red") +
  labs(title = "Elipsoide de nivel al 50%", x = "x_1", y = "x_2") +
  theme(plot.title = element_text(hjust=0.5))


library(MASS)  # Para la función cov.rob
library(ggplot2)

# Datos
x <- c(2, 8, 6, 8)
y <- c(12, 9, 9, 10)
df <- data.frame(x = x, y = y)

# Calcular la media y la matriz de covarianza
mean_vector <- colMeans(df)
cov_matrix <- cov(df)

# Calcular la distancia de Mahalanobis
mahal_dist <- mahalanobis(df, center = mean_vector, cov = cov_matrix)

# p-values de la distribución chi-cuadrado con 2 grados de libertad
p_values <- 1 - pchisq(mahal_dist, df = 2)

# Prueba de normalidad sobre las distancias usando Kolmogorov-Smirnov
ks_test <- ks.test(mahal_dist, "pchisq", df=2)

# Mostrar resultados
print(mahal_dist)  # Distancias de Mahalanobis
print(p_values)    # p-values individuales
print(ks_test)     # Prueba de normalidad Kolmogorov-Smirnov

# Gráfica QQ para evaluar normalidad de las distancias
ggplot(data.frame(mahal_dist), aes(sample = mahal_dist)) +
  stat_qq(distribution = qchisq, dparams = list(df=2)) +
  stat_qq_line(distribution = qchisq, dparams = list(df=2), color="red") +
  ggtitle("QQ-Plot de la Distancia de Mahalanobis") +
  xlab("Cuantiles Teóricos (Distribución Chi2)") +
  ylab("Distancias de Mahalanobis") +
  theme(plot.title = element_text(hjust=0.5))


### Calculamos la lambda de Wilks
n <- 4
p <- 2
mu_0 <- c(7,11)

x_1 <- c(2,8,6,8)
x_2 <- c(12,9,9,10)

X <- matrix(cbind(x_1,x_2), n, p)

P <- diag(rep(1,n)) - (1/n)*matrix(1,n,n)

S <- t(X) %*% P %*% X
S_total <- matrix(0, p, p) 
# Sumar (x_j - μ_0)(x_j - μ_0)^T para cada j
for (j in 1:nrow(X)) {
  diff <- as.matrix(X[j, ] - mu_0)  # Vector columna (p x 1)
  S_total <- S_total + diff %*% t(diff)  # Producto exterior (p x p)
}
lambda <- (det(S) / det(S_total))^(n/2)
sprintf("Lambda de Wilks = %.4f", lambda)




##### Datos radiacion

# Cargar librerías
library(readxl)
library(ggplot2)
library(ellipse)

# Leer los datos
radiacion <- read_excel("./datos_radiacion.xlsx")

# Transformar los datos (raíz cuarta)
radiacion_transformada <- data.frame(x1 = radiacion[[1]]^(1/4), 
                                     x2 = radiacion[[2]]^(1/4))
colnames(radiacion_transformada) <- c("x1", "x2")

n <- nrow(radiacion_transformada)
p <- ncol(radiacion_transformada)


# Calcular matriz de covarianza y media muestral
S <- cov(radiacion_transformada)  # Matriz de covarianza muestral
x_bar <- colMeans(radiacion_transformada)  # Media muestral



# Generar elipse basada en n(\bar{x} - \mu)^T S^{-1} (\bar{x} - \mu)
c <- sqrt((n-1)*p/(n-p)*qf(0.95,p,n-p))
d_elipse <- as.data.frame(ellipse(S/n, centre = x_bar, t=c ))
colnames(d_elipse) <- c("x1", "x2")

# Graficar la elipse con la media muestral
ggplot() +
  geom_path(data = d_elipse, aes(x = x1, y = x2), color = "red") + # Elipse
  geom_point(aes(x = x_bar[1], y = x_bar[2]), color = "blue", size = 3) + # Media
  labs(title = "Elipse de confianza al 95% para mu", x =  "mu_1", y = "mu_2") +
  #annotate("text", x = x_bar[1]+0.03, y = x_bar[2], label = "x̄") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5))


ggplot() +
  geom_path(data = d_elipse, aes(x = x1, y = x2), color = "red") + # Elipse
  geom_point(aes(x = x_bar[1], y = x_bar[2]), color = "blue", size = 3) + # Media
  labs(title = "Elipse de confianza al 95% para μ", x =  "μ_1", y = "μ_2") +
  geom_point(aes(x = 0.562, y = 0.589), color = "green", size = 3) +
  annotate("text", x = x_bar[1]+0.003, y = x_bar[2], label = "x̄") +
  annotate("text", x = 0.559, y = 0.589, label = "μ'") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5))

ggplot() +
  geom_path(data = d_elipse, aes(x = x1, y = x2), color = "red") + # Elipse
  geom_point(aes(x = x_bar[1], y = x_bar[2]), color = "blue", size = 3) + # Media
  labs(title = "Elipse de confianza al 95% para μ", x =  "μ_1", y = "μ_2") +
  geom_point(aes(x = 0.55,y = 0.60), color = "green", size = 3) +
  annotate("text", x = x_bar[1]+0.003, y = x_bar[2], label = "x̄") +
  annotate("text", x = 0.547, y = 0.60, label = "μ'") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5))


## INcios c


c <- sqrt((n-1)*p/(n-p)*qf(0.95,p,n-p))
d_elipse <- as.data.frame(ellipse(S/n, centre = x_bar, t = c))
eigvals_S <- eigen(S/n)

eje_1 <- eigvals_S$vectors[,1]*c*sqrt(eigvals_S$values[1])

eje_2 <- eigvals_S$vectors[,2]*c*sqrt(eigvals_S$values[2])

ggplot() +
  geom_segment(aes(x = x_bar[1]-eje_1[1], xend = x_bar[1] + eje_1[1],
                   y = x_bar[2]-eje_1[2], yend = x_bar[2] + eje_1[2]), color = "green") +
  geom_segment(aes(x = x_bar[1]-eje_2[1], xend = x_bar[1] + eje_2[1],
                   y = x_bar[2]-eje_2[2], yend = x_bar[2] + eje_2[2]), color = "cyan") +
  geom_path(data = d_elipse, aes(x = x1, y = x2), color = "red") + # Elipse
  geom_point(aes(x = x_bar[1], y = x_bar[2]), color = "blue", size = 3) + # Media
  labs(title = "Elipse de confianza al 95% para μ", x =  "μ_1", y = "μ_2") +
  annotate("text", x = x_bar[1]+0.0035, y = x_bar[2], label = "x̄") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5)) +
  coord_fixed()

mu_0 <- c(0.55,0.60)
S_inv <- solve(S)
t2 <- n*t(x_bar-mu_0) %*% S_inv %*% (x_bar-mu_0)
t2



#### Problema 4
mu_0 <- c(0.55,0.60)
a <- S_inv%*% (x_bar - mu_0)

y_bar <- t(a) %*% x_bar

Y <- data.frame(as.matrix(radiacion_transformada) %*% a)
colnames(Y) = c("y")

t <- (y_bar - t(a) %*% mu_0)/sqrt(t(a)%*%S%*%a/n)
t^2



#### Problema 5
library(readxl)
osos <- read_excel("./datos_osos.xlsx")
x_bar <- colMeans(osos)
S <- cov(osos)
s <- diag(S)

n <- nrow(osos)
p <- ncol(osos)

# inciso a
alfa <- 0.05
critico <- (n-1)*p/(n-p)*qf(alfa,p,n-p,lower.tail = FALSE)

intervalo_inf <- x_bar - sqrt(critico*s/n)
intervalo_sup <- x_bar + sqrt(critico*s/n)

intervalos <- data.frame(cbind(intervalo_inf, intervalo_sup))

library(knitr)
kable(intervalos)


# inciso b
# Obtenemos primero estos aumentos en las medias

# de 2 a 3
a <- as.matrix(c(-1,1,0,0), ncol=p)
diferencias <- diff(x_bar)
s <- t(a) %*% S %*% a
intervalo_inf <- diferencias[1] - sqrt(critico*s/n)
intervalo_sup <- diferencias[1] + sqrt(critico*s/n)
sprintf("Intervalo mu_3-mu_2 : (%.4f, %.4f)",intervalo_inf, intervalo_sup)
intervalo_2a3 <- c(intervalo_inf, intervalo_sup)


# de 3 a 4
a <- as.matrix(c(0,-1,1,0), ncol=p)
s <- t(a) %*% S %*% a
intervalo_inf <- diferencias[2] - sqrt(critico*s/n)
intervalo_sup <- diferencias[2] + sqrt(critico*s/n)
sprintf("Intervalo mu_4-mu_3 : (%.4f, %.4f)",intervalo_inf, intervalo_sup)
intervalo_3a4 <- c(intervalo_inf, intervalo_sup)

# de 4 a 5
a <- as.matrix(c(0,0,-1,1), ncol=p)
s <- t(a) %*% S %*% a
intervalo_inf <- diferencias[3] - sqrt(critico*s/n)
intervalo_sup <- diferencias[3] + sqrt(critico*s/n)
sprintf("Intervalo mu_5-mu_4 : (%.4f, %.4f)",intervalo_inf, intervalo_sup)
intervalo_4a5 <- c(intervalo_inf, intervalo_sup)

### Inciso c)

A <- matrix(c(-1,1,0,0,
              0,0,-1,1), nrow=2, byrow=TRUE)

S_dif <- A %*% S %*% t(A)
centro <- c(diferencias[1], diferencias[3])

p <- 4
t <- sqrt((n-1) * p / (n - p) * qf(alfa, p, n - p, lower.tail=FALSE))

library(ellipse)
elipse_data <- as.data.frame(ellipse(S_dif / n, centre = centro, t = t))

library(ggplot2)
ggplot() +
  geom_segment(aes(x = intervalo_2a3[1], xend=intervalo_2a3[1],
                   y = -26, yend = intervalo_4a5[2]), linetype = "dashed", linewidth = 0.3) +
  geom_segment(aes(x = intervalo_2a3[2], xend=intervalo_2a3[2],
                   y = -26, yend = intervalo_4a5[2]), linetype = "dashed", linewidth = 0.3) +
  geom_segment(aes(x = -27, xend=intervalo_2a3[2],
                   y = intervalo_4a5[1], yend = intervalo_4a5[1]), linetype = "dashed") +
  geom_segment(aes(x = -27, xend=intervalo_2a3[2],
                   y = intervalo_4a5[2], yend = intervalo_4a5[2]), linetype = "dashed") +
  geom_path(data = elipse_data, aes(x=x,y=y), color = "red")+
  xlim(intervalo_2a3*1.3) +
  ylim(intervalo_4a5*1.3) +
  labs(x = "Diferencia en longitud media de 2 a 3 años",y="Diferencia en longitud media de 4 a 5 años",
       title = "Intervalos de confianza simúltaneos al 95%") +
  theme_classic()
  

### Inciso d)
n_conjuntos <- 7
s <- diag(S)
alfa_i <- alfa / n

critical <- qt(alfa_i/2,n-1, lower.tail = FALSE)

critical / sqrt(p*(n-1)/(n-p)*qf(alfa,p,n-p, lower.tail = FALSE))


# Intervalos para las medias
t_intervalo_inf <- x_bar - critical*sqrt(s/n)
t_intervalo_sup <- x_bar + critical*sqrt(s/n)


# Intervalo para las diferencias longitudes medias anuales
tdif_intervalo_inf <- rep(0,3)
tdif_intervalo_sup <- rep(0,3)

a <- matrix(c(-1,1,0,0), nrow = 4, byrow = TRUE)
s <- t(a) %*% S %*% a 
tdif_intervalo_inf[1] <- t(a) %*% x_bar - critical * sqrt(s/n)
tdif_intervalo_sup[1] <- t(a) %*% x_bar + critical * sqrt(s/n)


a <- matrix(c(0,-1,1,0), nrow = 4, byrow = TRUE)
s <- t(a) %*% S %*% a 
tdif_intervalo_inf[2] <- t(a) %*% x_bar - critical * sqrt(s/n)
tdif_intervalo_sup[2] <- t(a) %*% x_bar + critical * sqrt(s/n)

a <- matrix(c(0,0,-1,1), nrow = 4, byrow = TRUE)
s <- t(a) %*% S %*% a 
tdif_intervalo_inf[3] <- t(a) %*% x_bar - critical * sqrt(s/n)
tdif_intervalo_sup[3] <- t(a) %*% x_bar + critical * sqrt(s/n)

