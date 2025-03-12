contaminantes <- read.table("./data/T1-5.dat")

names(contaminantes) <- c("viento", "radiacion", "co","no","no2","o3","hc")

## Inciso a)

modelo_a <- lm(no2 ~ viento + radiacion + co + no + hc, data = contaminantes)
summary(modelo_a)

### Con un \alpha = 0.5 y realizando diferentes pruebas de hipotesis para
### los coeficientes de viento, radiacion, no, y o3 obtenemos que no se 
### obtiene evidencia suficiente para concluir que sus coeficientes sea distinto
### de cero.

modelo_acortado <- lm(no2 ~ co + hc, data=contaminantes)
summary(modelo_acortado)

residuos <- resid(modelo_acortado)
par(mar=c(4,4,2,2))
plot(residuos, xlab = "Índice de Observación", ylab = "Residuo")
text(x = 1:length(residuos), y = residuos, labels = 1:length(residuos), pos = 4, cex = 0.8, col = "black")
abline(h = 0, col = "red", lty = 2)

qqpoints <- qqnorm(residuos)
qqline(residuos)
cor(qqpoints$x, qqpoints$y)

plot(modelo_acortado, which=1)
plot(modelo_acortado, which=2)
plot(modelo_acortado, which=3)

### Realizar un análisis de regresión lineal utilizando Y1=no2 y Y2=o3

modelo_2 <- lm(cbind(no2, o3) ~ viento + radiacion + co+ no+hc , data = contaminantes)
summary(modelo_2)
