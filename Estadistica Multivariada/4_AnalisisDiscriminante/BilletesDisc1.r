# clear all variables
rm(list = ls(all = TRUE))
graphics.off()

# reads the bank data
x = read.table("bank2.dat")

xg  = x[1:100, ]                # Grupo de las primeras 100 observaciones    
xf  = x[101:200, ]              # Grupo de las segundas 100 observaciones 
mg  = colMeans(xg)              # Determinar la media de los grupos por separado y la muestra total
mf  = colMeans(xf)
m   = (mg + mf)/2
w   = 100 * (cov(xg) + cov(xf)) # Matriz w de la suma de cuadrados
d   = mg - mf                   # Diferencia de medias
a   = solve(w) %*% d            # Determinar los factores por combinaciones lineales 

yg = as.matrix(xg - matrix(m, nrow = 100, ncol = 6, byrow = T)) %*% a  # Regla disciriminante para billetes genuinos 
yf = as.matrix(xf - matrix(m, nrow = 100, ncol = 6, byrow = T)) %*% a  # Regla Discriminante para billetes falsos 

xgtest = yg
sg = sum(xgtest < 0)            # Numero de billetes genuinos mal clasificados

xftest = yf                     # Numero de billetes falsos mal clasificados 
sf = sum(xftest > 0)

fg = density(yg)                # Densidad de la proyeccion de billetes genuinos 
ff = density(yf)                # Densidad de la proyeccions de billetes falsos 

# plot
plot(ff, lwd = 3, col = "red", xlab = "", ylab = "Densidades de Proyecciones", main = "Densidades de proyecciones de los billetes suizos", 
     xlim = c(-0.2, 0.2), cex.lab = 1.2, cex.axis = 1.2, cex.main = 1.8)
lines(fg, lwd = 3, col = "blue", lty = 2)
text(mean(yf), 3.72, "Falsificado", col = "red")
text(mean(yg), 2.72, "Genuino", col = "blue")

## Utilizando discriminante lineal
dataBilletes = data.frame(Status=c(rep("G",100), rep("F",100)), x)

lda_billetes=lda(Status~.,data=dataBilletes)
Predicciones=predict(lda_billetes,dataBilletes)
table(Predicciones$class, dataBilletes$Status)
lda_billetes

## Con opcion
lda_cv_billetes=lda(Status~.,data=dataBilletes,CV=TRUE)

# las predicciones ya estan generadas en  lda_cv_billetes
table(lda_cv_billetes$class, dataBilletes$Status)

# Graficar las predicciones - Primer discriminador lineal
ldahist(data = Predicciones$x[,1], g=dataBilletes$Status)


library(klaR)
dataBilletes[,"Status"] = as.factor(dataBilletes[,"Status"])
partimat(Status~.,data=dataBilletes,method="lda")

