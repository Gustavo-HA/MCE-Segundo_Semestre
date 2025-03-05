#GEN, genero (hombre = 0, mujer = 1)
#AMT, Cantidad de droga tomada al momento de la sobredosis
#PR, Medicion de la onda PR
#DIAP, Presion sanguinea diastolica
#QRS, Medicion de la onda QRS
## TOT Nivel de plasma TCAD
## AMI Cantidad de droga Amitriptilina en el plasma


## Regresion lineal multiple
ami_data <- read.table("ami_data.DAT")
names(ami_data) <- c("TOT","AMI","GEN","AMT","PR","DIAP","QRS")

summary(ami_data)
pairs(ami_data)

mlm1 <- lm(cbind(TOT, AMI) ~ GEN + AMT + PR + DIAP + QRS, data = ami_data)
summary(mlm1)
## los resultados de cada una de las regresiones corresponde a una regresion
## univariada sobre cada uno de los regresores

head(resid(mlm1))
head(fitted(mlm1))
coef(mlm1)
sigma(mlm1)

## Verificacion de la matriz de varianzas y covarianzas del modelo
vcov(mlm1)


## Verificacion de la influencias de los predictores
library(car)
Anova(mlm1)

## Verificacion con un modelo reducido
mlm2 <- update(mlm1, . ~ . - PR - DIAP - QRS)
summary(mlm2)

anova(mlm1, mlm2)

lh.out <- linearHypothesis(mlm1, hypothesis.matrix = c("PR = 0", "DIAP = 0", "QRS = 0"))
lh.out

## Estadistico de Wilks
E <- lh.out$SSPE
H <- lh.out$SSPH
det(E)/det(E + H)

## Formula de pillai
sum(diag(H %*% solve(E + H)))

## Hotelling-Lawley
sum(diag(H %*% solve(E)))

## Estadistico de Roy
e.out <- eigen(H %*% solve(E))
max(e.out$values)

## Nos quedamos con un modelo reducido
summary(mlm2)

anova(mlm2)

## Realizar una prediccion para TOT y AMI con los genero=1 y AMT=1200
nd <- data.frame(GEN = 1, AMT = 1200)
p <- predict(mlm2, nd)
p

## Generacion de una elipse de prediccion
predictionEllipse <- function(mod, newdata, level = 0.95, ggplot = TRUE){
  # labels
  lev_lbl <- paste0(level * 100, "%")
  resps <- colnames(mod$coefficients)
  title <- paste(lev_lbl, "confidence ellipse for", resps[1], "and", resps[2])

  # prediction
  p <- predict(mod, newdata)

  # center of ellipse
  cent <- c(p[1,1],p[1,2])

  # shape of ellipse
  Z <- model.matrix(mod)
  Y <- mod$model[[1]]
  n <- nrow(Y)
  m <- ncol(Y)
  r <- ncol(Z) - 1
  S <- crossprod(resid(mod))/(n-r-1)

  # radius of circle generating the ellipse
  tt <- terms(mod)
  Terms <- delete.response(tt)
  mf <- model.frame(Terms, newdata, na.action = na.pass,
                    xlev = mod$xlevels)
  z0 <- model.matrix(Terms, mf, contrasts.arg = mod$contrasts)
  rad <- sqrt((m*(n-r-1)/(n-r-m))*qf(level,m,n-r-m)*z0%*%solve(t(Z)%*%Z) %*% t(z0))

  # generate ellipse using ellipse function in car package
  ell_points <- car::ellipse(center = c(cent), shape = S, radius = c(rad), draw = FALSE)

  # ggplot2 plot
  if(ggplot){
    require(ggplot2, quietly = TRUE)
    ell_points_df <- as.data.frame(ell_points)
    ggplot(ell_points_df, aes(x, y)) +
      geom_path() +
      geom_point(aes(x = TOT, y = AMI), data = data.frame(p)) +
      labs(x = resps[1], y = resps[2],
           title = title)
  } else {
    # base R plot
    plot(ell_points, type = "l", xlab = resps[1], ylab = resps[2], main = title)
    points(x = cent[1], y = cent[2])
  }
}

predictionEllipse(mod = mlm2, newdata = nd)

## Ejemplo usando IRIS
library("car")
some(iris)

## Visualizacion de datoshttp://127.0.0.1:23507/graphics/plot_zoom_png?width=1164&height=706
scatterplotMatrix(~ Sepal.Length + Sepal.Width + Petal.Length
                  + Petal.Width | Species,
                  data=iris, smooth=FALSE, regLine=FALSE, ellipse=TRUE,
                  by.groups=TRUE, diagonal=FALSE, legend=list(coords="bottomleft"))

## Grafica de caja de los datos
par(mfrow=c(2, 2))
for (response in c("Sepal.Length", "Sepal.Width",
                   "Petal.Length", "Petal.Width"))
  Boxplot(iris[, response] ~ Species, data=iris, ylab=response)

## Modelo multivariado de especies
mod.iris <- lm(cbind(Sepal.Length, Sepal.Width, Petal.Length, Petal.Width)
               ~ Species, data=iris)

## Verificacion del objeto
class(mod.iris)
mod.iris
summary(mod.iris)

## Comparacion con una anova
## Hipotesis: todas las especies son iguales en sus cuatro medidas
(manova.iris <- Anova(mod.iris))
class(manova.iris)
summary(manova.iris)

anova(mod.iris)



