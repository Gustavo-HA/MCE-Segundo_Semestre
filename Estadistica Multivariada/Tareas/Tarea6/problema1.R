library("ca")
library(ggplot2)


tabla <- matrix(c(10,0,0,0,
                  7,12,0,0,
                  0,5,15,0,
                  0,0,23,19), nrow=4, byrow=TRUE)

chisq.test(tabla)
ancor <- ca(tabla, nd=2)

row_cord <- ancor$rowcoord
col_cord <- ancor$colcoord

row_names <- c("M1", "M2", "M3", "M4")
col_names <- c("H1", "H2", "H3", "H4")


df_rows <- data.frame(
  Dim1 = row_cord[, 1],
  Dim2 = row_cord[, 2],
  Label = row_names,
  Type = "Filas (Mujer)"
)

df_cols <- data.frame(
  Dim1 = col_cord[, 1],
  Dim2 = col_cord[, 2],
  Label = col_names,
  Type = "Columnas (Hombre)"
)

plot_data <- rbind(df_rows, df_cols)

ca_plot_ggplot <- ggplot(plot_data, aes(x = Dim1, y = Dim2, color = Type, shape = Type)) +
  geom_point(size = 4) +
  geom_text(aes(label = Label), nudge_y = 0.15, size = 3, show.legend = FALSE) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "grey50") +
  geom_vline(xintercept = 0, linetype = "dashed", color = "grey50") +
  scale_color_manual(values = c("Filas (Mujer)" = "blue", "Columnas (Hombre)" = "red")) +
  scale_shape_manual(values = c("Filas (Mujer)" = 19, "Columnas (Hombre)" = 19)) +
  labs(
    x = paste0("Dim 1 (64.8%)"),
    y = paste0("Dim 2 (26.8%)"),
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

