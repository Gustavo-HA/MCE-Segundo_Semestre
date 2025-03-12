# Conjunto de datos para ejemplo
# paquete CCA para test y graficas
library(CCA)

### EJEMPLO 1
### Ahorro en el ciclo de vida
data(LifeCycleSavings)
head(LifeCycleSavings,10)

# Correr correlacion canocia
pop <- LifeCycleSavings[, 2:3]    # seleccionar pop15 and pop75
oec <- LifeCycleSavings[, -(2:3)] # seleccionar los dem?s
cancor(pop, oec)

cor(LifeCycleSavings[,-1])


matcor(pop,oec)

res.cc=cc(pop,oec)

par(mar=c(4,4,2,2))
plt.cc(res.cc,type="i")

# paquete yacca
library(yacca)
options(scipen=999)
cca.fit = cca(pop,oec)
F.test.cca(cca.fit)


## EJEMPLO 2
## Variables academicas
mm = read.csv("https://stats.idre.ucla.edu/stat/data/mmreg.csv")
colnames(mm) = c("Control", "Concept", "Motivation", "Read", "Write", "Math", "Science","Sex")

# Primeras 10 lineas
head(mm,10)

# Creacion de la matrices X y Y
psych = mm[, 1:3]
acad = mm[, 4:7]

# Matrices de correlacion Rxx, Ryy y  Rxy
matcor(psych,acad)

# Correr analisis de correlacion
library(CCA)

cc1 = cc(psych,acad)
cc1

# Graficar primera dimension
plt.cc(cc1,type="v",var.label=TRUE)
plt.cc(cc1)


# Calcular las cargas de correlacion canonica
cc2 = comput(psych,acad,cc1)
cc2

# Calcular el test F de las variables canonicas
library(yacca)
options(scipen=999)
cca2.fit = cca(psych,acad)
F.test.cca(cca2.fit)

# Calcular los coeficientes canonicos

#  variables psicologicas
psychsd =  diag(sqrt(diag(cov(psych))))
Xout = matrix(psychsd%*%cc1$xcoef,nrow=3,ncol=3,byrow=FALSE,dimnames = list(c("control","concept","motivation"),c("CV1","CV2","CV3")))

# variables  Academicas
acadsd = diag(sqrt(diag(cov(acad))))
Yout = matrix(acadsd%*%cc1$ycoef,nrow=4,ncol=3,byrow=FALSE,dimnames = list(c("read","write","math","science"),c("CV1","CV2","CV3")))

