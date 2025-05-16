

# Practica 5. Aplicacion de an?lisis de correspondencia en R
library("MASS")
#library("alr3")
library("stats")
library("scatterplot3d")
library("smacof")
library("ca")
#source("/OTRAS ACTIVIDADES ROD/curso_INEGI_AGS_MULTIVARIADO/analisis_factores/functions_imp.r")

######################################################################################


##analisis de correspondencia

#vamos a realizar un analisis de correspondencia para representar las filas y las columnas en un espacio conjunto de baja dimension


#primer tabla de contingencia
#la tabla presenta la clasificacion de 5387 escolares escoceses por el color de sus ojos que tiene cuatro categorias: claros,azules,casta?os y oscuros (I=4)
# y por el color de su cabello que tiene 5 categorias:rubio, pelirrojo,casta?o,oscuro y negro (J=5).
#La tabla tiene interes historico porque fue usada por Fisher en 1940 para ilustrar otro metodo de analisis tablas de contingencia.


# Se introduce la tabla
fisher_tab<-matrix(c(688,116,584,188,4,326,38,241,110,3,343,84,909,412,26,98,48,403,681,85),ncol=5,byrow=TRUE) 
fisher_tab

n<-sum(fisher_tab)

#se calcula la chi cuadrada para probar la independencia de los renglones y columnas de la tabla de contingencia y la inercia total (medida de la
#variabilidad total de los datos en la tabla)

chiR<-chisq.test(fisher_tab)
inetot<-chiR$statistic/n



# Se calcula la matriz F de frecuencias relativas o matriz de correspondencia
nrow<-nrow(fisher_tab)
ncol<-ncol(fisher_tab)
F_fisher<-(fisher_tab)/n



#se calculan las frecuencias relativas de las filas y de las columnas de F o tambien llamadas masas (las ri y las cj)
rtot<-apply(F_fisher,1,sum)
ctot<-apply(F_fisher,2,sum)
rtot
ctot

# los vectores obtenidos se  colocan en las matrices diagonales correspondientes (Dr y Dc)
Dr<-diag(rtot)
Dc<-diag(ctot)


# se calcula la matriz de perfiles por fila (R)
table.pro<-F_fisher/rtot  #matriz R de renglones


# se calcula la matriz de perfiles por columna (Rc)
table.pcol<- t(t(F_fisher)/ctot)  #matriz R de columnas


#se calcula la chi cuadrada para probar la independencia de los renglones y columnas de la tabla de contingencia y la inercia total (medida de la
#variabilidad total de los datos en la tabla)

t.rowsum<-n*rtot #suma de las entradas por fila de la tabla de contingencia
t.colsum<-n*ctot #suma de las entradas por columna de la tabla de contingencia  
tabla_exp<-t.rowsum%o%t.colsum/n  # matriz de frecuencias esperadas
tabla_exp
chi2<- sum((fisher_tab-tabla_exp)^2/tabla_exp)
chi2
inetot1<-chi2/n
inetot1



#se calcula la matriz Z
Z<-(sqrt(solve(Dr)))%*%F_fisher%*%(sqrt(solve(Dc)))

#se obtienen los vectores propios y valores propios mediante la descomposicion en valores singulares de Z
dvalsing<-svd(Z)



#Se obtienen las representaciones de las filas y columnas en un espacio de dos dimensiones considerando los vectores propios ai y bi obtenidos de la dvs
#OJO:se consideran solo los vectores propios asociados a valores propios distintos de 1, en este caso se consideran las columnas 2 y 3 de las matrices U y V 
#de la dvs.
ind<-c(2,3)
#Representacion de las filas en dos dimensiones
Cr<-(sqrt(solve(Dr)))%*%Z%*%dvalsing$v[,ind]
Cr



#Representacion de las columnas en dos dimensiones
Cc<-(sqrt(solve(Dc)))%*%t(Z)%*%dvalsing$u[,ind]
Cc

#se calcula la proporcion de la inercia explicada por las dos  dimensiones (asociadas a valores propios mas grandes distintos de uno)
vp<-(dvalsing$d)^2
vp_dist1<-vp[-1]
vp_dist1
cumsum(vp_dist1) / sum(vp_dist1)
var_expl <- sum(vp_dist1[1:2])/sum(vp_dist1)
var_expl

# Se obtiene la representacion conjunta de los renglones y columnas en el espacio de dos dimensiones

par(pty="s")

plot(Cr,xlim=range(Cr[,1],Cc[,1]),ylim=range(Cr[,1],Cc[,1]),
     xlab="Coordenada 1",ylab="Coordenada 2",lwd=2)
points(Cc,col=2)
#plot(BB2[,1],BB2[,2],xlim=range(AA2[,1],BB2[,1]),ylim=range(AA2[,1],BB2[,1]),type="n",
#    xlab="Coordenada 1",ylab="Coordenada 2",lwd=2)
text(Cr,labels=c("o claros","o azules","o castanos","o oscuros"),col=1,lwd=2)
text(Cc,labels=c("c rubio","c pelirrojo","c castanos","c oscuro","c negro"),col=2,lwd=4)

abline(h=0,lty=2)
abline(v=0,lty=2)


#Se calculan las distancias chi cuadrada de cada renglon de R a su centroide: (Ra-c)'Dc^-1(Ra-c)
dist.chicua.ren<-sqrt(apply((t(table.pro)-ctot)^2/ctot,2,sum))
dist.chicua.ren
#se calcula la inercia de cada renglon, que se obtiene simplemente multiplicando las distancia chicuadrada a su centroide (al cuadrado) por su masa correspondiente
inercia.ren<-((dist.chicua.ren)^2) *rtot
inercia.ren

#Se calculan las distancias chi cuadrada de cada columna de R a su centroide: (Ra-r)'Dr^-1(Ra-r)
dist.chicua.col<-sqrt(apply(((table.pcol)-rtot)^2/rtot,2,sum))
dist.chicua.col
#se calcula la inercia de cada columna, que se obtiene simplemente multiplicando las distancia chicuadrada a su centroide (al cuadrado) por su masa correspondiente
inercia.col<-((dist.chicua.col)^2) *ctot
inercia.col

sum(inercia.ren)
sum(inercia.col)

#ahora vamos usar la funcion "ca" de R para realizar analisis de correspondencia sobre la misma tabla de contingencia
#nota: esta funcion trabaja directamente con  la matriz F-Fesp, la cual ya no tiene un valor propio de 1. 

corres<-ca(fisher_tab, nd = 2)
corres<-ca(fisher_tab, nd = 2, suprow = NA, supcol = NA, 
           subsetrow = NA, subsetcol = NA)



Cr1<-corres$rowcoord #coordenadas de las filas
Cc1<-corres$colcoord  #coordenadas de las columnas

# Se obtiene la representacion conjunta de los renglones y columnas en el espacio de dos dimensiones

par(pty="s")

plot(Cr1,xlim=range(Cr1[,1],Cc1[,1]),ylim=range(Cr1[,2],Cc1[,2]),
     xlab="Coordenada 1",ylab="Coordenada 2",lwd=2)
points(Cc1,col=2)
#plot(BB2[,1],BB2[,2],xlim=range(AA2[,1],BB2[,1]),ylim=range(AA2[,1],BB2[,1]),type="n",
#    xlab="Coordenada 1",ylab="Coordenada 2",lwd=2)
text(Cr1,labels=c("o claros","o azules","o castanos","o oscuros"),col=1,lwd=2)
text(Cc1,labels=c("c rubio","c pelirrojo","c castanos","c oscuro","c negro"),col=2,lwd=4)

abline(h=0,lty=2)
abline(v=0,lty=2)


#conclusion: el gr?fico muestra  una relaci?n clara entre ambas variables, es decir, existe dependencia entre las dos variables. La dimension principal gradua la tonalidad
#de claro a oscuro y la segunda separa los casta?os de los demas


############################################################################################################################3
#otro ejemplo que esta dentro la libreria ca


fumadores<-data(smoke)


n<-sum(smoke)

#se calcula la chi cuadrada para probar la independencia de los renglones y columnas de la tabla de contingencia y la inercia total (medida de la
#variabilidad total de los datos en la tabla)

chiR<-chisq.test(smoke)
inetot<-chiR$statistic/n


#se aplica analisis de correspondencia mediante la funcion ca de R para obtener una reprsentacion conjunta de filas y columnas

corres_fum<-ca(smoke)
Cr<-corres_fum$rowcoord
Cc<-corres_fum$colcoord
#plot(corres_fum)
#text(Cr,labels=rownames(smoke),col=1,lwd=2)
#text(Cc,labels=colnames(smoke),col=2,lwd=4)

# Se obtiene la representacion conjunta de los renglones y columnas en el espacio de dos dimensiones

par(pty="s")

plot(Cr,xlim=range(Cr[,1],Cc[,1]),ylim=range(Cr[,2],Cc[,2]),
     xlab="Coordenada 1",ylab="Coordenada 2",lwd=2)
points(Cc,col=2)
#plot(BB2[,1],BB2[,2],xlim=range(AA2[,1],BB2[,1]),ylim=range(AA2[,1],BB2[,1]),type="n",
#    xlab="Coordenada 1",ylab="Coordenada 2",lwd=2)
text(Cr,labels=rownames(smoke),col=1,lwd=2)
text(Cc,labels=colnames(smoke),col=2,lwd=4)

abline(h=0,lty=2)
abline(v=0,lty=2)


#en las siguientes lineas se realiza el analisis de corresponencia a mano

nrow<-nrow(smoke)
ncol<-ncol(smoke)
n<-sum(smoke)

#se calcula la chi cuadrada para probar la independencia de los renglones y columnas de la tabla de contingencia y la inercia total (medida de la
#variabilidad total de los datos en la tabla)

chiR<-chisq.test(smoke)
inetot<-chiR$statistic/n

# Se calcula la matriz F de frecuencias relativas o matriz de correspondencia

F_smoke<-(smoke)/n


#se calculan las frecuencias relativas de las filas y de las columnas de F  o tambien llamadas masas (las ri y las cj)
rtot<-(apply(smoke,1,sum))/n
ctot<-apply(smoke,2,sum)/n


# los vectores obtenidos se  colocan en las matrices diagonales correspondientes (Dr y Dc)
Dr<-diag(rtot)
Dc<-diag(ctot)


# se calcula la matriz de perfiles por fila (R)
table.pro<-F_smoke/rtot  #matriz R de renglones

# se calcula la matriz de perfiles por columna (Rc)
table.pcol<- t(t(F_smoke)/ctot)  #matriz R de columnas


#se calcula la chi cuadrada y la inercia total

t.rowsum<-n*rtot
t.colsum<-n*ctot
tabla_exp<-t.rowsum%o%t.colsum/n
chi2<- sum((smoke-tabla_exp)^2/tabla_exp)
chi2
inetot<-chi2/n
inetot


#se calcula la matriz Z
Z1<-(sqrt(solve(Dr)))%*%as.matrix(F_smoke)%*%(sqrt(solve(Dc)))


#se obtienen los vectores propios y valores propios mediante la descomposicion en valores singulares de Z
dvalsing<-svd(Z1)


#Se obtienen las representaciones de las filas y columnas en un espacio de dos dimensiones considerando los vectores propios ai y bi obtenidos de la dvs
#OJO:se consideran solo los vectores propios asociados a valores propios distintos de 1, en este caso se consideran las columnas 2 y 3 de las matrices U y V 
#de la dvs.
ind<-c(2,3)
#Representacion de las filas en dos dimensiones
Cr<-(sqrt(solve(Dr)))%*%Z1%*%dvalsing$v[,ind]
Cr

#Representacion de las columnas en dos dimensiones
Cc<-(sqrt(solve(Dc)))%*%t(Z1)%*%dvalsing$u[,ind]
Cc

#se calcula la proporcion de la inercia explicada por las dos  dimensiones (asociadas a valores propios distintos de cero)
vp<-(dvalsing$d)^2
vp_dist1<-vp[-1]
vp_dist1
var_expl <- sum(vp_dist1[1:2])/sum(vp_dist1)


# Se obtiene la representacion conjunta de los renglones y columnas en el espacio de dos dimensiones

par(pty="s")

plot(Cr,xlim=range(Cr[,1],Cc[,1]),ylim=range(Cr[,1],Cc[,1]),
     xlab="Coordenada 1",ylab="Coordenada 2",lwd=2)
points(Cc,col=2)
#plot(BB2[,1],BB2[,2],xlim=range(AA2[,1],BB2[,1]),ylim=range(AA2[,1],BB2[,1]),type="n",
#    xlab="Coordenada 1",ylab="Coordenada 2",lwd=2)
text(Cr,labels=rownames(smoke),col=1,lwd=2)
text(Cc,labels=colnames(smoke),col=2,lwd=4)

abline(h=0,lty=2)
abline(v=0,lty=2)


#Se calculan las distancias chi cuadrada de cada renglon de R a su centroide: (Ra-c)'Dc^-1(Ra-c)
dist.chicua.ren<-sqrt(apply((t(table.pro)-ctot)^2/ctot,2,sum))
dist.chicua.ren
#se calcula la inercia de cada renglon, que se obtiene simplemente multiplicando las distancia chicuadrada a su centroide (al cuadrado) por su masa correspondiente
inercia.ren<-((dist.chicua.ren)^2) *rtot
inercia.ren

#Se calculan las distancias chi cuadrada de cada columna de R a su centroide: (Ra-c)'Dr^-1(Ra-c)
dist.chicua.col<-sqrt(apply(((table.pcol)-rtot)^2/rtot,2,sum))
dist.chicua.col
#se calcula la inercia de cada columna, que se obtiene simplemente multiplicando las distancia chicuadrada a su centroide (al cuadrado) por su masa correspondiente
inercia.col<-((dist.chicua.col)^2) *ctot
inercia.col


#interpretacion de la representacion















