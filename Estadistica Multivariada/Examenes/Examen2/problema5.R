# Problema 5

library("ca")
library("ggplot2")
table <- matrix(c(121,57,72,36,21,
                  288,105,141,97,71,
                  112,65,77,54,54,
                  86,60,94,78,71), nrow=4, byrow=TRUE)

n <- 1660
chiR <- chisq.test(table)

ancor <- ca(table)


row_cord <- ancor$rowcoord
col_cord <- ancor$colcoord

row_names <- c("Well", "MildSF", "ModSF", "Impaired")
col_names <- c("A", "B", "C", "D", "E")

cumsum(ancor$sv**2) / sum(ancor$sv**2)
ancor$sv**2 / sum(ancor$sv**2)

# Visualizacion de resultados.
df_rows <- data.frame(
  Dim1 = row_cord[, 1],
  Dim2 = row_cord[, 2],
  Label = row_names,
  Type = "Filas"
)

df_cols <- data.frame(
  Dim1 = col_cord[, 1],
  Dim2 = col_cord[, 2],
  Label = col_names,
  Type = "Columnas"
)


plot_data <- rbind(df_rows, df_cols)

ca_plot_ggplot <- ggplot(plot_data, aes(x = Dim1, y = Dim2, color = Type, shape = Type)) +
  geom_point(size = 4) +
  geom_text(aes(label = Label), nudge_y = 0.15, size = 3, show.legend = FALSE) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "grey50") +
  geom_vline(xintercept = 0, linetype = "dashed", color = "grey50") +
  scale_color_manual(values = c("Filas" = "blue", "Columnas" = "red")) +
  scale_shape_manual(values = c("Filas" = 19, "Columnas" = 19)) +
  labs(
    x = paste0("Dim 1 (89.79%)"),
    y = paste0("Dim 2 (7.49%)"),
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





