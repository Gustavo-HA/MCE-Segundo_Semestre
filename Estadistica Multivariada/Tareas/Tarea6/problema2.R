library("ca")

tabla <- matrix(c(42,62,184, 207,
                  13,28, 81, 113,
                  7,18, 54, 92), nrow=3, byrow=TRUE)

tabla

n <- sum(tabla)
row_names <- c("L", "M", "H")
col_names <- c("VD", "SD", "MS", "VS")

# Prueba chi-cuadrado y análisis de correspondencia
chi_test <- chisq.test(tabla)
chi_test$statistic/n

ancor <- ca(tabla, nd=2)
row_cord <- ancor$rowcoord
col_cord <- ancor$colcoord

cumsum(ancor$sv**2) / sum(ancor$sv**2) # Si se puede usar una dimensión
ancor$sv**2 / sum(ancor$sv**2)

# Visualizacion de resultados.
df_rows <- data.frame(
  Dim1 = row_cord[, 1],
  Dim2 = row_cord[, 2],
  Label = row_names,
  Type = "Filas (Salario)"
)

df_cols <- data.frame(
  Dim1 = col_cord[, 1],
  Dim2 = col_cord[, 2],
  Label = col_names,
  Type = "Columnas (Satisfaccion)"
)


plot_data <- rbind(df_rows, df_cols)

ca_plot_ggplot <- ggplot(plot_data, aes(x = Dim1, y = Dim2, color = Type, shape = Type)) +
  geom_point(size = 4) +
  geom_text(aes(label = Label), nudge_y = 0.15, size = 3, show.legend = FALSE) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "grey50") +
  geom_vline(xintercept = 0, linetype = "dashed", color = "grey50") +
  scale_color_manual(values = c("Filas (Salario)" = "blue", "Columnas (Satisfaccion)" = "red")) +
  scale_shape_manual(values = c("Filas (Salario)" = 19, "Columnas (Satisfaccion)" = 19)) +
  labs(
    x = paste0("Dim 1 (99.02%)"),
    y = paste0("Dim 2 (0.98%)"),
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

