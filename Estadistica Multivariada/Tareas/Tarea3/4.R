vinos <- read.csv("./data/wine.data", header = FALSE)
names(vinos) <- c("Class","Alcohol","Malic", "Ash", "Alcal", "Mg",
                  "Phenol","Flav","Nonf","Proan","Color","Hue","Abs","Proline")
str(vinos)
unique(vinos$Class)

vinos$Class <- as.factor(vinos$Class)

library(MASS)


## Inciso a)
lda_modelo <- lda(Class ~ ., data = vinos)
lda_modelo

predichos <- predict(lda_modelo)$class
table(Prediccion = predichos, Real = vinos$Class)

## Inciso b)
predichos <- predict(lda_modelo)
df_plot <- data.frame(
  LD1 = predichos$x[,1],
  LD2 = predichos$x[,2],
  Class = vinos$Class
)

library(ggplot2)
ggplot(data=df_plot, aes(x = LD1, y = LD2, color=Class, shape=Class)) +
  geom_point(size=3) +
  labs(title="Observaciones en el espacio de discriminantes")


## Inciso c)
### Validación cruzada
cv_modelo_lda <- lda(Class ~ ., data=vinos, CV=TRUE)

confusion <- table(Prediccion = cv_modelo_lda$class, Real = vinos$Class)
confusion


### Train-Test
set.seed(123)  # Semilla para reproducibilidad
train_index <- sample(seq_len(nrow(vinos)), size = 0.8 * nrow(vinos))
train_data <- vinos[train_index, ]
test_data  <- vinos[-train_index, ]

modelo_lda <- lda(Class ~ ., data = train_data)
modelo_lda

predicciones <- predict(modelo_lda, newdata = test_data)
tabla_conf <- table(Predicho = predicciones$class, Real = test_data$Class)
tabla_conf

exactitud <- mean(predicciones$class == test_data$Class)
exactitud



