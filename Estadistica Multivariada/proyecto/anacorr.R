library("ca")
library("ggplot2")

data <- read.csv("./data/bank-full.csv", sep = ";")

position_marital <- data[, c("marital", "job")]

# Create a contingency table
tabla <- table(position_marital)

chisq.test(tabla)

ancor<- ca(table(position_marital))

row_cord <- ancor$rowcoord
col_cord <- ancor$colcoord

row_names <- rownames(row_cord)
col_names <- rownames(col_cord)

cumsum(ancor$sv**2) / sum(ancor$sv**2) # Si se puede usar una dimensión
ancor$sv**2 / sum(ancor$sv**2)

# Visualizacion de resultados.
df_rows <- data.frame(
  Dim1 = row_cord[, 1],
  Dim2 = row_cord[, 2],
  Label = row_names,
  Type = "Filas (marital)"
)

df_cols <- data.frame(
  Dim1 = col_cord[, 1],
  Dim2 = col_cord[, 2],
  Label = col_names,
  Type = "Columnas (type of job)"
)


plot_data <- rbind(df_rows, df_cols)

ca_plot_ggplot <- ggplot(plot_data, aes(x = Dim1, y = Dim2, color = Type, shape = Type)) +
  geom_point(size = 4) +
  geom_text(aes(label = Label), nudge_y = 0.15, size = 3, show.legend = FALSE) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "grey50") +
  geom_vline(xintercept = 0, linetype = "dashed", color = "grey50") +
  scale_color_manual(values = c("Filas (marital)" = "blue", "Columnas (type of job)" = "red")) +
  scale_shape_manual(values = c("Filas (marital)" = 19, "Columnas (type of job)" = 19)) +
  labs(
    x = paste0("Dim 1 (89.4%)"),
    y = paste0("Dim 2 (10.6%)"),
    title = "Análisis de Correspondencia",
    color = "Tipo",
    shape = "Tipo"
  ) +
  theme_minimal(base_size = 11) +
  theme(
    plot.title = element_text(hjust = 0.5),
    legend.position = "none"
  )

print(ca_plot_ggplot)
