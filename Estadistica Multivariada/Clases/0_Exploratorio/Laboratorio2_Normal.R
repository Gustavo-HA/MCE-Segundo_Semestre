## Author:       Wolfgang Haerdle 20091001
##

##
## 1 Verificacion Normal Bivariada##
##

# Librerias
#install.packages("MASS")
#install.packages("mnormt")
library(MASS)
library(mnormt)

# Parametros en entrada
n     = 200                                 # numero de muestras
mu    = c(3,2)                              # vector de medias
sig   = matrix(c(1, -1.5, -1.5, 4), ncol=2) # matriz de covarianzas

# Muestreo normal bivariada
set.seed(80)
y     = mvrnorm(n, mu, sig, 2)

# Densidad Normal Bivariada
xgrid = seq( from=(mu[1]-3*sqrt(sig[1,1])),to=(mu[1]+3*sqrt(sig[1,1])),length.out=200 )
ygrid = seq( from=(mu[2]-3*sqrt(sig[2,2])),to=(mu[2]+3*sqrt(sig[2,2])),length.out=200 )
z     = outer( xgrid,ygrid,FUN=function(xgrid,ygrid){dmnorm(cbind(xgrid,ygrid),mean=mu,varcov=sig)} )

# Plot
par(mfrow=c(1,2))

# Grafica dispersion
plot(y, col="black", ylab="X2",xlab="X1",xlim=range(xgrid),ylim=range(ygrid))
title('Muestra Normal')

# Elipses de contorno
contour(xgrid,ygrid,z,xlim=range(xgrid),ylim=range(ygrid),nlevels=10,col = c("blue", "black", "yellow", "cyan", "red", "magenta", "green", "blue", "black"), lwd=3, cex.axis = 1, xlab="X1", ylab="X2")
title('Elipses de contorno')

##
## 2 Limite central
##

#Librerias
## install.packages("KernSmooth")
library(KernSmooth)

p=0.5
n=5

bsample = rbinom(n*1000, 1, 0.5)                       #Generacion aleatoria  de la distribucion binomial con parametros 1000*n y 0.5
bsamplem = matrix(bsample, n, 1000)                    #Creacion de una matrix de variables binomiales aleatorias
bden = bkde((colMeans(bsamplem) - p)/sqrt(p*(1-p)/n))  #Estimar la densidad del kernel

plot(bden, col = "blue3", type="l", lty = 1, lwd=4, xlab="1000 muestras aleatorias",
ylab="Densidad Normal y estimada", cex.lab=1, cex.axis = 1, ylim=c(0, 0.45))     #Graficar la densidad del kernel
lines(bden$x, dnorm(bden$x),col = "red3", lty = 1, lwd=4)                        #Graficar la densidad normal

title(paste("Distribucion Asintotica, n =",n))

##
p=0.5
n=6

bsample = rbinom(n*2000, 1, 0.5)          #Generacion aleatoria de la distribucion binomal con parametros 2000*n and 0.5
bsamplem = matrix(bsample, n, 2000)       #Crear una matriz de las variables aleatorias binomiales
bsamplemstd = matrix((colMeans(bsamplem) - p)/sqrt(p*(1-p)/n), 1000, 2)

dj = bkde2D(bsamplemstd, bandwidth = 1.06*c(sd(bsamplemstd[, 1]), sd(bsamplemstd[, 2]))* 200^(-1/5))  #Calculo de los kernel bidimensionales

persp(dj$x1, dj$x2,dj$fhat, box=FALSE, theta = 265, phi = 15, r = sqrt(3), d = 1,ltheta = -135, lphi = 0, shade = NA)

