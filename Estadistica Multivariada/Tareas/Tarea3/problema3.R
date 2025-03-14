library(robustbase)
vaso
head(vaso)
str(vaso)

vaso$Y <- as.factor(vaso$Y)

# Inciso a)

library(ggplot2)

ggplot(data=vaso)+
  geom_point(aes(x=Volume, y=Rate, color=Y)) +
  theme(legend.position = "none")

## analisis discriminante
library(MASS)
modelo_lda <- lda(Y ~ Volume + Rate, data=vaso)

# Predicciones en los datos originales
predicciones_lda <- predict(modelo_lda)

# Ver primeras predicciones
head(predicciones_lda$class)  # Clase predicha
head(predicciones_lda$posterior)  # Probabilidades posteriores


# Matriz de confusión
tabla_confusion <- table(Predicho = predicciones_lda$class, Real = vaso$Y)
print(tabla_confusion)

# Calcular la precisión
precision <- sum(diag(tabla_confusion)) / sum(tabla_confusion)
cat("Accuracy del modelo LDA:", precision, "\n")
TP <- tabla_confusion[2,2]
FP <- sum(tabla_confusion[2,]) - TP
FN <- sum(tabla_confusion[,2]) - TP
TN <- sum(tabla_confusion) - TP - FN - FP
precision <- TP / (TP + FP)
recall <- TP / (TP + FN)
cat("Precisión:", precision, "\n")
cat("Recall   :", recall, "\n")


# Agregar la proyección a los datos
vaso$LD1 <- predicciones_lda$x[,1]  # Primera función discriminante

# Visualización de los grupos en el espacio discriminante
ggplot(vaso, aes(x = LD1, fill = Y)) +
  geom_histogram(alpha = 0.5, position = "identity", bins = 20) +
  labs(title = "Distribución de Clases en el Espacio Discriminante")

# Generar una grilla de valores
grid <- expand.grid(
  Volume = seq(min(vaso$Volume), max(vaso$Volume), length.out = 100),
  Rate = seq(min(vaso$Rate), max(vaso$Rate), length.out = 100)
)

# Predecir la clase para cada punto de la grilla
grid$Y_pred <- predict(modelo_lda, newdata = grid)$class

# Graficar puntos originales y frontera de decisión
ggplot(vaso) +
  geom_point(size = 2, aes(x = Volume, y = Rate, color = Y)) +  # Scatter plot of actual data points
  geom_tile(data = grid, aes(x = Volume, y = Rate, fill = Y_pred), alpha = 0.3) +
  scale_fill_manual(values = c("red", "blue")) +
  labs(title = "Frontera de Decisión LDA", fill = "Clase Predicha") +
  theme(legend.position="none")


# 2. Fit QDA model
modelo_qda <- qda(Y ~ Volume + Rate, data = vaso)

# 3. Predictions on the original data
predicciones_qda <- predict(modelo_qda)

# 4. Confusion matrix
tabla_confusion_qda <- table(Predicho = predicciones_qda$class, Real = vaso$Y)
print(tabla_confusion_qda)

# 5. Calculate accuracy
precision_qda <- sum(diag(tabla_confusion_qda)) / sum(tabla_confusion_qda)
cat("Accuracy del modelo QDA:", precision_qda, "\n")
TP <- tabla_confusion_qda[2,2]
FP <- sum(tabla_confusion_qda[2,]) - TP
FN <- sum(tabla_confusion_qda[,2]) - TP
TN <- sum(tabla_confusion_qda) - TP - FN - FP
precision <- TP / (TP + FP)
recall <- TP / (TP + FN)
cat("Precisión:", precision, "\n")
cat("Recall   :", recall, "\n")


# 6. Create a grid for decision boundary
grid <- expand.grid(
  Volume = seq(min(vaso$Volume), max(vaso$Volume), length.out = 100),
  Rate   = seq(min(vaso$Rate),   max(vaso$Rate),   length.out = 100)
)

# 7. Predict class for each point in the grid
grid$Y_pred_qda <- predict(modelo_qda, newdata = grid)$class

# 8. Plot original points + QDA decision boundary
ggplot(vaso) +
  geom_point(size = 2, aes(x = Volume, y = Rate, color = Y)) +    # actual data
  geom_tile(data = grid, aes(x = Volume, y = Rate, fill = Y_pred_qda), alpha = 0.3) +
  scale_fill_manual(values = c("red", "blue")) +
  labs(title = "Frontera de Decisión QDA", fill = "Clase Predicha") +
  theme(legend.position = "none")


# Inciso b)

library(robustbase)
vaso[,1] <- log(vaso[,1])
vaso[,2] <- log(vaso[,2])

vaso$Y <- as.factor(vaso$Y)

# Inciso a)

library(ggplot2)

ggplot(data=vaso)+
  geom_point(aes(x=Volume, y=Rate, color=Y)) +
  theme(legend.position = "none")

## analisis discriminante
library(MASS)
modelo_lda <- lda(Y ~ Volume + Rate, data=vaso)

# Predicciones en los datos originales
predicciones_lda <- predict(modelo_lda)

# Ver primeras predicciones
head(predicciones_lda$class)  # Clase predicha
head(predicciones_lda$posterior)  # Probabilidades posteriores


# Matriz de confusión
tabla_confusion <- table(Predicho = predicciones_lda$class, Real = vaso$Y)
print(tabla_confusion)

# Calcular la precisión
precision <- sum(diag(tabla_confusion)) / sum(tabla_confusion)
cat("Accuracy del modelo LDA:", precision, "\n")

TP <- tabla_confusion[2,2]
FP <- sum(tabla_confusion[2,]) - TP
FN <- sum(tabla_confusion[,2]) - TP
TN <- sum(tabla_confusion) - TP - FN - FP
precision <- TP / (TP + FP)
recall <- TP / (TP + FN)
cat("Precisión:", precision, "\n")
cat("Recall   :", recall, "\n")



# Agregar la proyección a los datos
vaso$LD1 <- predicciones_lda$x[,1]  # Primera función discriminante

# Visualización de los grupos en el espacio discriminante
ggplot(vaso, aes(x = LD1, fill = Y)) +
  geom_histogram(alpha = 0.5, position = "identity", bins = 20) +
  labs(title = "Distribución de Clases en el Espacio Discriminante")

# Generar una grilla de valores
grid <- expand.grid(
  Volume = seq(min(vaso$Volume), max(vaso$Volume), length.out = 100),
  Rate = seq(min(vaso$Rate), max(vaso$Rate), length.out = 100)
)

# Predecir la clase para cada punto de la grilla
grid$Y_pred <- predict(modelo_lda, newdata = grid)$class

# Graficar puntos originales y frontera de decisión
ggplot(vaso) +
  geom_point(size = 2, aes(x = Volume, y = Rate, color = Y)) +  # Scatter plot of actual data points
  geom_tile(data = grid, aes(x = Volume, y = Rate, fill = Y_pred), alpha = 0.3) +
  scale_fill_manual(values = c("red", "blue")) +
  labs(title = "Frontera de Decisión LDA", fill = "Clase Predicha") +
  theme(legend.position="none")


# 2. Fit QDA model
modelo_qda <- qda(Y ~ Volume + Rate, data = vaso)

# 3. Predictions on the original data
predicciones_qda <- predict(modelo_qda)

# 4. Confusion matrix
tabla_confusion_qda <- table(Predicho = predicciones_qda$class, Real = vaso$Y)
print(tabla_confusion_qda)

# 5. Calculate accuracy
precision_qda <- sum(diag(tabla_confusion_qda)) / sum(tabla_confusion_qda)
cat("Accuracy del modelo QDA:", precision_qda, "\n")
TP <- tabla_confusion_qda[2,2]
FP <- sum(tabla_confusion_qda[2,]) - TP
FN <- sum(tabla_confusion_qda[,2]) - TP
TN <- sum(tabla_confusion_qda) - TP - FN - FP
precision <- TP / (TP + FP)
recall <- TP / (TP + FN)
cat("Precisión:", precision, "\n")
cat("Recall   :", recall, "\n")


# 6. Create a grid for decision boundary
grid <- expand.grid(
  Volume = seq(min(vaso$Volume), max(vaso$Volume), length.out = 100),
  Rate   = seq(min(vaso$Rate),   max(vaso$Rate),   length.out = 100)
)

# 7. Predict class for each point in the grid
grid$Y_pred_qda <- predict(modelo_qda, newdata = grid)$class

# 8. Plot original points + QDA decision boundary
ggplot(vaso) +
  geom_point(size = 2, aes(x = Volume, y = Rate, color = Y)) +    # actual data
  geom_tile(data = grid, aes(x = Volume, y = Rate, fill = Y_pred_qda), alpha = 0.3) +
  scale_fill_manual(values = c("red", "blue")) +
  labs(title = "Frontera de Decisión QDA", fill = "Clase Predicha") +
  theme(legend.position = "none")
  

