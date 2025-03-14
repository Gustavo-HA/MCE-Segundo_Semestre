contaminantes <- read.table("./data/T1-5.dat")

names(contaminantes) <- c("viento", "radiacion", "co","no","no2","o3","hc")

## Inciso a)

modelo_a <- lm(no2 ~ viento + radiacion, data = contaminantes)
summary(modelo_a)
residuos <- resid(modelo_a)
shapiro.test(residuos)
par(mar=c(4,4,2,2))
plot(modelo_a)

### Realizar un análisis de regresión lineal utilizando Y1=no2 y Y2=o3

  modelo_2 <- lm(cbind(no2, o3) ~ viento + radiacion , data = contaminantes)
  summary(modelo_2)
  plot(modelo_2)


