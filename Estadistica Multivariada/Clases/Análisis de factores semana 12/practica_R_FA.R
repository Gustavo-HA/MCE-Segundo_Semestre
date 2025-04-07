

# Practica 4. Aplicacion de analisis de factores a un conjunto de datos con el fin de reducir la dimensionalidad de los datos
library("MASS")
#library("alr3")
library("stats")
library("scatterplot3d")
library("psych") #prueba de esfericidad de bartlett
#source("/OTRAS ACTIVIDADES ROD/curso_INEGI_AGS_MULTIVARIADO/analisis_factores/functions_imp.r")
library(dtw)
######################################################################################

#Primer conjunto de datos 

#Los datos consisten de las caracter?sticas de veh?culos de diferentes marcas  y modelos.
#El objetivo principal del estudio es predecir las ventas de autom?viles a partir del conjunto de variables.
#Sin embargo, algunas de estas variables est?n correlacionadas, lo cual  puede afectar desfavorablemente
#los resultados de la predicci?n.

#Las variables a considerar son:
# Tipo de veh?culo (camioneta o auto compacto)
#Precio del veh?culo
#Tama?o del motor
#Caballos de fuerza
#Distancia entre ejes
#Ancho
#Longitud
#Peso neto
#Capacidad de combustible
#Consumo o rendimiento



#DATOS_AF<- read.csv("DATOS_AF.csv", header = FALSE, sep = ",", quote = "\"",dec = ".", fill = TRUE, na.strings = "NA")


ventas_coche <- read.csv("car_sales.csv", header = TRUE, sep = ",", quote = "\"",dec = ".", fill = TRUE, na.strings = "NA")
#datos_ven <- read.csv("datos_vendedores.csv", header = TRUE, sep = ",", quote = "\"",dec = ".", fill = TRUE, na.strings = "NA")

#extraemos los nombre de variables
lista<- names(ventas_coche)


#se extraen los nombres de los coches (casos)
nombres_coches<-ventas_coche[,1]

# se extraen solo las variables de interes
var_coches <-ventas_coche[,5:14]
n<-nrow(var_coches)

# Se revisa la correlaci?n entre las variables para ver si vale la pena realizar analisis de factores
varcoches_corr <-cor(var_coches)
det_vc<-det(varcoches_corr)
det_vc # Las variables parecen ser corelacionadas


#Prueba de esfericidad de bartlett para probar la hipotesis nula de que las variables
#no estan correlacionadas. La idea es rechazar la hipotesis nula para proseguir con un
#analisis de factores

#cortest.bartlett(var_coches) #funcion que realiza la prueba de esfericidad de Bartlett
cortest.bartlett(varcoches_corr,n) # H_0 : Sigma = I

#Se realiza el analisis de factores utilizando componentes principales para estimar las cargas

coches.pc<-princomp(var_coches, cor=TRUE)
summary(coches.pc)
screeplot(coches.pc)

# por lo tanto elegimos un modelo con m=2 ? m=3 factores como m?ximo, de acuerdo a
#los criterios

#se obtienen las cargas (matriz L)
cargas_coches<-coches.pc$loadings
cargas_coches

#elegimos las primeras 3 columnas de la matriz de carga(L), correspondiente a m=3
cargas_3comp<-cargas_coches[,1:3]
cargas_3comp

#calcular las cargas como sqrt(lamda)*vector propio

#se calculan las comunalidades para cada variable (hi^2)
# hi^2 = diagonal de LL' 
comun<-diag((cargas_3comp%*%t(cargas_3comp)))
comun

#se calculan las varianzas especificas para cada variable (las psi)
varesp<- (1-comun)
varesp


#se obtiene la estimacion de la matriz de correlaciones (matriz reproducida)
# LL' + psi
pred3_vc<- cargas_3comp%*%t(cargas_3comp)+diag(varesp)
pred3_vc

varcoches_corr #matriz de correlaci?n real(observada)

#se calcula la diferencia entre la matriz de correlacion observada y la matriz reproducida
#(matriz de residuales)
round(varcoches_corr-pred3_vc,digits=3)

#tambien se puede aplicar la funcion prcomp para obtener las cargas por componentes principales
#con esta funcion se pueden rotar las cargas
coches.pc1<-prcomp(var_coches, retx = TRUE, center = TRUE, scale. = TRUE)

cargas_3comp1<-coches.pc1$rotation #matriz de cargas rotadas

screeplot(coches.pc1)


#elegimos las primeras 3 columnas de la matriz de cargas rotada (L*), correspondiente a m=3
cargas_3comp_rotadas<-cargas_3comp1[,1:3]
cargas_3comp_rotadas



#el siguiente paso seria intepretar los factores en terminos de sus cargas

#Nota: no es facil interpretar los factores de acuerdo a los valores de cargas, porque
#todas las cargas de los factores tienen valores similares en las variables
#aun cuando las cargas estan rotadas



# Se realiza un analisis de factores utilizando maxima verosimilitud para estimar los parametros del modelo
#(las cargas y las varianzas especificas).



#se prueba la solucion con un factor (m=1)
venta_coche.fa1<- factanal(var_coches,factors=1) #funcion que realiza el analisis de factores mediante maxima verosimilitud
#Por default el analisis de factores se realiza sobre los datos estandarizados z y utilizando la rotaci?n varimax

CARGAS1<-venta_coche.fa1$loadings #cargas estimadas
VAR_ESP1<-venta_coche.fa1$uniquenesses #varianzas especificas estimadas
prueba_hipo1<-venta_coche.fa1$PVAL #prueba de hipotesis para determinar si un factor es adecuado

CARGAS1
VAR_ESP1
prueba_hipo1
comun1<-1-VAR_ESP1
comun1


#se prueba la solucion con dos factores (m=2)
venta_coche.fa2<- factanal(var_coches,factors=2) #funcion que realiza el analisis de factores
#var_vend.fa2<- factanal(var_vend,factors=2)
CARGAS2<-venta_coche.fa2$loadings #cargas estimadas
VAR_ESP2<-venta_coche.fa2$uniquenesses #varianzas especificas estimadas
prueba_hipo2<-venta_coche.fa2$PVAL #prueba de hipotesis para determinar si dos factores es adecuado

CARGAS2
VAR_ESP2
prueba_hipo2
comun2<-1-VAR_ESP2
comun2


#se prueba la solucion con tres factores factores (m=3)
venta_coche.fa3<- factanal(var_coches,factors=3) #funcion que realiza el analisis de factores
CARGAS3<-venta_coche.fa3$loadings #cargas estimadas
VAR_ESP3<-venta_coche.fa3$uniquenesses #varianzas especificas estimadas
prueba_hipo3<-venta_coche.fa3$PVAL #prueba de hipotesis para determinar si tres factores es adecuado

CARGAS3
VAR_ESP3
prueba_hipo3
comun3<-1-VAR_ESP3
comun3

#se prueba la solucion con cuatro factores factores (m=4)
venta_coche.fa4<- factanal(var_coches,factors=4) #funcion que realiza el analisis de factores
CARGAS4<-venta_coche.fa4$loadings #cargas estimadas
VAR_ESP4<-venta_coche.fa4$uniquenesses #varianzas especificas estimadas
prueba_hipo4<-venta_coche.fa4$PVAL #prueba de hipotesis para determinar si cuatro factores es adecuado

CARGAS4
VAR_ESP4
prueba_hipo4
comun4<-1-VAR_ESP4
comun4

#se prueba la solucion con cuatro factores factores (m=5)
venta_coche.fa5<- factanal(var_coches,factors=5) #funcion que realiza el analisis de factores
CARGAS5<-venta_coche.fa5$loadings #cargas estimadas
VAR_ESP5<-venta_coche.fa5$uniquenesses #varianzas especificas estimadas
prueba_hipo5<-venta_coche.fa5$PVAL #prueba de hipotesis para determinar si cuatro factores es adecuado

CARGAS5
VAR_ESP5
prueba_hipo5
comun5<-1-VAR_ESP5
comun5


#se prueba la solucion con cuatro factores factores (m=6)
venta_coche.fa6<- factanal(var_coches,factors=6) #funcion que realiza el analisis de factores
CARGAS6<-venta_coche.fa6$loadings #cargas estimadas
VAR_ESP6<-venta_coche.fa6$uniquenesses #varianzas especificas estimadas
prueba_hipo6<-venta_coche.fa6$PVAL #prueba de hipotesis para determinar si cuatro factores es adecuado

CARGAS6
VAR_ESP6
prueba_hipo6 # Ya no se cumple la condición sobre m.
comun6<-1-VAR_ESP6
comun6


#Se calcula la diferencia entre las correlaciones observadas y las predichas
#para m=2 factores

#Primero se obtiene la estimacion de la matriz de correlaciones (matriz reproducida)
pred2_vc<- CARGAS2%*%t(CARGAS2)+diag(VAR_ESP2)

#se calcula la diferencia entre la matriz de correlacion observada y la matriz reproducida
round(varcoches_corr-pred2_vc,digits=3)


#Se calcula la diferencia entre las correlaciones observadas y las predichas para m=3 factores

#Primero se obtiene la estimacion de la matriz de correlaciones (matriz reproducida)
pred3_vc <-CARGAS3%*%t(CARGAS3)+diag(VAR_ESP3)

#se calcula la diferencia entre la matriz de correlacion observada y la matriz reproducida

round(varcoches_corr-pred3_vc,digits=3)


# por lo anterior el modelo de 3 factores produce una matriz residual cercana a cero. Por tanto
#un modelo con tres factores puede ser suficiente para explicar la variabilidad de los datos originales


#Interpretaci?n de los factores de acuerdo a los valores de sus cargas

CARGAS3


#El factor 1 se relaciona al precio del coche, al tama?o del motor y a
#los caballos de fuerza, principalmente con los caballos de fuerza
#(factor relacionado a la potencia del coche y al precio)

#El factor 2 se relaciona a la distancia entre ejes,anchura y longitud del
#coche (principalmente con longitud) (factor relacionado al tama?o del coche)

#El factor 3 se relaciona a basicamente al tipo de vehiculo (camioneta o un
#coche compacto) y a la capacidad de combustible (factor relacionado al tipo de vehiculo)


#se calculan los factor scores con el modelo factorial para m=3

#Obs:cuando se deseen obtener los factor scores, se debe utilizar como datos
#de entrada la matriz original de datos y no la matriz de covarianzas

factor_coches_reg <- factanal(var_coches,factors=3,scores="regression")
scores_reg<-factor_coches_reg$scores
scores_reg

factor_coches_ms <- factanal(var_coches,factors=3,scores="Bartlett")
scores_ms<-factor_coches_reg$scores
scores_ms

scores_reg[1,]
scores_ms[1,]

#calcula el producto de (L'*(Psi^-1)*L)^-1=DELTA^-1 

solve(t(CARGAS3)%*%solve(diag(VAR_ESP3))%*%CARGAS3)


#se grafican los 117 casos de acuerdo a los factor scores obtenidos con regresion
scatterplot3d(scores_reg, angle=35, col.grid="lightblue", main="Grafica de los factor scores", pch=20)


#se a?aden los nombres de los coches en los factor scores 
dimnames(scores_reg)[[1]]<-nombres_coches

#vamos a graficar los factor scores tomados dos a dos


#f1 x f2
par(pty="s")
plot(scores_reg[,1],scores_reg[,2],
     ylim=range(scores_reg[,1]),
     xlab="Factor 1",ylab="Factor 2",type="n",lwd=2)
text(scores_reg[,1],scores_reg[,2],
     labels=abbreviate(row.names(scores_reg),minlength=8),cex=0.6,lwd=2)


#f1 x f3
par(pty="s")
plot(scores_reg[,1],scores_reg[,3],
     ylim=range(scores_reg[,1]),
     xlab="Factor 1",ylab="Factor 3",type="n",lwd=2)
text(scores_reg[,1],scores_reg[,3],
     labels=abbreviate(row.names(scores_reg),minlength=8),cex=0.6,lwd=2)


#f2 x f3
par(pty="s")
plot(scores_reg[,2],scores_reg[,3],
     ylim=range(scores_reg[,2]),
     xlab="Factor 2",ylab="Factor 3",type="n",lwd=2)
text(scores_reg[,2],scores_reg[,3],
     labels=abbreviate(row.names(scores_reg),minlength=8),cex=0.6,lwd=2)

#el factor 1 es el que est? influyendo en el comportamiento fuera de rango del
#coche


#Conclusiones

# 1.La dimension de los datos se redujo de 10 variables a tres factores aplicando an?lisis
#factorial con el metodo de estimacion de MV. 
# 2. La interpretaci?n de los factores depende de las relaciones definidas por la
#matriz de cargas rotadas
# 3. La reduccion de la dimension de los datos y el uso de los factor scores
#(no correlacionados) como variables predictoras en el modelo de regresion,
#podrian tener beneficio en el pronostico de las ventas de coches
#y en analisis posteriores.
###############################################################################################################################



#Segundo conjunto de datos 



telcom<- read.csv("telco.csv", header = TRUE, sep = ",", quote = "\"",dec = ".", fill = TRUE, na.strings = "NA")

#nombre de variables
lista<- names(telcom)

#se identifican las variables (servicios) de interes para el analisis
var_int<- c(16, 17, 18, 19, 20, 26, 27, 28, 29, 30, 31, 32, 33, 34)

# se extraen solo las variables de interes
telcom1 <-telcom[,var_int]


# Se eliminan las filas con NA utilizando la funci?n na.omit()
telcom1<-na.omit(telcom1)


# Se revisa la correlaci?n entre las variables para ver si vale la pena realizar analisis de factores
telcom1_corr <-cor(telcom1)
det_telcom<-det(telcom1_corr)

#Prueba de esfericidad de bartlett para probar la hipotesis nula de que las variables
#no estan correlacionadas. La idea es rechazar la hipotesis nula para proseguir con un
#analisis de factores

cortest.bartlett(telcom1_corr)



# se realiza un analisis de factores (utilizando MV en la estimacion), probando  de 1 a 6 factores y considerando una rotacion varimax
telcom_fa<- vector ("list",6)
for(i in 1:6){
  telcom_fa[[i]]<-  factanal(covmat=telcom1_corr,factors=i)
}

# Se considera la soluci?n con 4 factores como la adecuada

telcom.fa4<-factanal(telcom1,factors=4)


#Se calcula la diferencia entre las correlaciones observadas y las predichas para m=4 factores
pred4_telcom <-telcom.fa4$loadings%*%t(telcom.fa4$loadings)+diag(telcom.fa4$uniquenesses)

round(telcom1_corr-pred4_telcom,digits=3)


#Interpretaci?n de los factores

#El factor 1: Servicio gratis el mes pasado, Identificaci?n de llamadas, Llamadas en espera,desvio de llamada,Llamada de 3 simultaneamente
#El factor 2:  Equipamiento el mes pasado,internet, Facturaci?n electr?nica
#El factor 3 :Larga distancia el mes pasado
#el factor 4 no tiene una clara relacion con algunas variables en particular auqnue parece mas relacionadsa a Wireless el mes pasado


#Por lo tanto, se puede identificar tres grandes grupos de servicios, seg?n la definici?n de los servicios que est?n m?s altamente correlacionados
#con los cuatro factores. 

# Grupo1: grupo de servicios extras
# Grupo2: Grupo de servicios de tecnologia
#Grupo de larga distancia


