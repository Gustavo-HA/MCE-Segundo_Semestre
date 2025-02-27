# Cargar la libreria con la funcion lda() 
library(MASS)
# Guardamos el conjunto de datos
dataset=iris

# Documentacion de ayuda de la funcion
?lda

# Realizamos LDA sobre los datos
lda_iris=lda(Species~.,data=dataset)
# Probabilidades previas y  coeficientes de los discriminantes lineales
lda_iris

# Verificamos la precision del analisis
Predicciones=predict(lda_iris,dataset)
table(Predicciones$class, dataset$Species)

# Verificamos que tan facil es separar de manera lineal el conjunto de datos
pairs(dataset)

# LDA con CV
lda_cv_iris=lda(Species~.,data=dataset,CV=TRUE)

# las predicciones ya estan generadas en  lda_cv_iris
table(lda_cv_iris$class, dataset$Species)

# Graficar las predicciones - Primer discriminador lineal
ldahist(data = Predicciones$x[,1], g=dataset$Species)

#Graficar las predicciones - Segundo discriminador lineal
ldahist(data = Predicciones$x[,2], g=dataset$Species)

# Analisis discriminante cuadratico
qda_iris=qda(Species~.,data=dataset)
qda_iris

# Verificar la asertividad del qda
Predicciones_qda=predict(qda_iris,dataset)
table(Predicciones_qda$class, dataset$Species)

# Usar el paquete klarR 
# install.packages("klaR")
library(klaR)
partimat(Species~.,data=dataset,method="lda")
partimat(Species~.,data=dataset,method="qda")

mu.k <- lda_iris$means
mu <- colMeans(mu.k)
dscores <- scale(dataset[,1:4], center=mu, scale=F) %*% lda_iris$scaling

spid=as.numeric(iris$Species)
plot(dscores, xlab="LD1", ylab="LD2", pch=spid, col=spid,  main="Scores Discriminantes", xlim=c(-10, 10), ylim=c(-3, 3))
legend("top",lev,pch=1:3,col=1:3,bty="n")

species <- factor(rep(c("s","c","v"), each=50))  
partimat(x=dscores[,2:1], grouping=species, method="lda")

## Los scores  fueron obtenidos con el metodo lineal
partimat(x=dscores[,2:1], grouping=species, method="qda")

