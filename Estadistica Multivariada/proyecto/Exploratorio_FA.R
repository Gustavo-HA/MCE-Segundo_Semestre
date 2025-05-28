# ===============================
# LIMPIEZA DEL ENTORNO Y RUTAS
# ===============================
rm(list = ls())
library(corrplot)
# ===============================


# Analisis exploratorio del Dataset
# Cargar el dataset
bank <- read.csv("./data/bank-full.csv", sep = ";", header = TRUE, stringsAsFactors = TRUE)

# Ver la estructura del dataset
str(bank)
# Ver el resumen del dataset
summary(bank)

# ===============================
# ANÁLISIS DE FACTORES
# ===============================

# Seleccionar las variables numéricas que se incluirán en el análisis de factores
variables_factor <- bank[, c("age", "balance", "day", "duration", "campaign", "pdays", "previous")] # ¡SELECCIONA LAS VARIABLES RELEVANTES!

# 1. Verificar la adecuación de los datos
cor_matrix <- cor(variables_factor)
print("Matriz de Correlaciones:")
print(cor_matrix)

# Visualizamos la matriz de correlación
# Aumentar tamaño de la ventana
par(oma=c(1,1,1,1), mar=c(1,1,1,1))
corrplot(cor_matrix, method = "color", type = "upper", tl.col = "black", tl.cex = 0.8,
         tl.srt = 45, addCoef.col = "black", number.cex = 0.7)


# Prueba de Bartlett de Esfericidad
library(psych)
bartlett_test <- cortest.bartlett(cor_matrix, n = nrow(bank)) # n es el número de observaciones
print("Prueba de Bartlett de Esfericidad:")
print(bartlett_test)

# Medida de Adecuación Muestral Kaiser-Meyer-Olkin (KMO)
kmo_test <- KMO(cor_matrix)
print("Medida de Adecuación Muestral KMO:")
print(kmo_test)

# 2. Extracción de factores (ejemplo con componentes principales)
factor_analysis_pc <- principal(variables_factor, nfactors = 3, rotate = "varimax") # Especifica el número de factores y el método de rotación
print("Análisis de Factores (Componentes Principales):")
print(factor_analysis_pc)

# Gráfico de Sedimentación (Scree Plot) para ayudar a determinar el número de factores
plot(factor_analysis_pc$values, type = "b", main = "Gráfico de Sedimentación", xlab = "Número de Componentes", ylab = "Autovalor")
abline(h = 1, col = "red", lty = 2) # Línea para el criterio de Kaiser

# 3. Interpretación de los factores (a partir del output de 'principal')
# Examina las cargas factoriales (loadings) para ver qué variables se asocian más con cada factor.

# 4. Puntuaciones Factoriales (opcional)
factor_scores <- factor.scores(variables_factor, factor_analysis_pc)$scores
print("Puntuaciones Factoriales:")
print(head(factor_scores))


# 5. Visualización de las cargas factoriales
library(ggplot2)

# Convertimos a data.frame para usar ggplot
factor_scores_df <- as.data.frame(factor_scores)

# Graficamos RC1 vs RC2
ggplot(factor_scores_df, aes(x = RC1, y = RC2)) +
  geom_point(alpha = 0.6, color = "#0072B2") +
  geom_hline(yintercept = 0, linetype = "dashed", color = "gray50") +
  geom_vline(xintercept = 0, linetype = "dashed", color = "gray50") +
  labs(
    title = "Proyección de individuos en el espacio factorial (RC1 vs RC2)",
    x = "RC1: Historial de contacto previo",
    y = "RC2: Intensidad de la campaña"
  ) +
  theme_minimal(base_size = 14)

# Graficamos RC1 vs RC3
ggplot(factor_scores_df, aes(x = RC1, y = RC3)) +
  geom_point(alpha = 0.6, color = "#D55E00") +
  geom_hline(yintercept = 0, linetype = "dashed", color = "gray50") +
  geom_vline(xintercept = 0, linetype = "dashed", color = "gray50") +
  labs(
    title = "Proyección de individuos en el espacio factorial (RC1 vs RC3)",
    x = "RC1: Historial de contacto previo",
    y = "RC3: Caracteristicas del cliente"
  ) +
  theme_minimal(base_size = 14)


# Graficamos RC2 vs RC3
ggplot(factor_scores_df, aes(x = RC2, y = RC3)) +
  geom_point(alpha = 0.6, color = "#009E73") +
  geom_hline(yintercept = 0, linetype = "dashed", color = "gray50") +
  geom_vline(xintercept = 0, linetype = "dashed", color = "gray50") +
  labs(
    title = "Proyección de individuos en el espacio factorial (RC2 vs RC3)",
    x = "RC2: Intensidad de la campaña",
    y = "RC3: Caracteristicas del cliente"
  ) +
  theme_minimal(base_size = 14)

# Visualización de las puntuaciones factoriales coloreadas por la variable objetivo
factor_scores_df$target <- bank$y  # o la variable que uses

ggplot(factor_scores_df, aes(x = RC1, y = RC2, color = target)) +
  geom_point(alpha = 0.6) +
  scale_color_manual(values = c("no" = "#D55E00", "yes" = "#009E73")) +
  labs(
    title = "Proyección factorial coloreada por variable objetivo",
    x = "RC1: Historial previo",
    y = "RC2: Intensidad campaña",
    color = "Respuesta"
  ) +
  theme_minimal(base_size = 14)

# Visualización 3D de las cargas factoriales coloradas por la variable objetivo
library(plotly)
plot_ly(
  factor_scores_df, 
  x = ~RC1, y = ~RC2, z = ~RC3,
  color = ~target,
  colors = c("no" = "#D55E00", "yes" = "#009E73"),
  type = 'scatter3d',
  mode = 'markers',
  marker = list(size = 3)
) %>%
  layout(title = "Puntuaciones factoriales (RC1, RC2, RC3) coloreadas por variable objetivo",
         scene = list(xaxis = list(title = "RC1: Historial de contacto previo"),
                      yaxis = list(title = "RC2: Intensidad de la campaña"),
                      zaxis = list(title = "RC3: Características del cliente")))

# Visualización 3D de las puntuaciones factoriales
library(plotly)

plot_ly(
  factor_scores_df, 
  x = ~RC1, y = ~RC2, z = ~RC3,
  type = 'scatter3d',
  mode = 'markers',
  marker = list(size = 3, color = ~RC3, colorscale = "Viridis")
) %>%
  layout(title = "Puntuaciones factoriales (RC1, RC2, RC3)")



# ===============================

# ANÁLISIS DE CLUSTERS
# ===============================
library(cluster)

# Usamos k-means con k = 2, ya que tú ya viste 2 grupos
set.seed(123)
clustering <- kmeans(factor_scores, centers = 2)

# Agregar el cluster al data.frame
factor_scores_df$cluster <- as.factor(clustering$cluster)

# Visualizar en 2D
ggplot(factor_scores_df, aes(x = RC1, y = RC2, color = cluster)) +
  geom_point(alpha = 0.6) +
  labs(
    title = "Clusters en espacio factorial (K-Means k=2)",
    x = "RC1", y = "RC2"
  ) +
  theme_minimal()


table(Cluster = clustering$cluster, Target = bank$y)
# 





