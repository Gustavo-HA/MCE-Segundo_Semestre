library("ca")
library("ggplot2")


perform_correspondence_analysis <- function(contingency_table,
                                          legend_1, legend_2, dims, nd) {
  
  # Perform Chi-squared test (optional, but good for checking association)
  print(chisq.test(contingency_table))
  
  # Perform Correspondence Analysis
  ca_results <- ca(contingency_table, nd=nd)
  
  # Extract row and column coordinates
  row_coord <- ca_results$rowcoord
  col_coord <- ca_results$colcoord
  
  row_names_ca <- rownames(row_coord)
  col_names_ca <- rownames(col_coord)
  
  # Calculate inertia (variance explained) for each dimension
  inertia <- ca_results$sv^2 / sum(ca_results$sv^2)
  inertia_dim1_pct <- round(inertia[dims[1]] * 100, 1)
  inertia_dim2_pct <- round(inertia[dims[2]] * 100, 1)
  
  # Cumulative inertia (optional, good for deciding number of dimensions)
  print(cumsum(ca_results$sv^2) / sum(ca_results$sv^2))
  
  # Prepare data for plotting
  df_rows <- data.frame(
    Dim1 = row_coord[, dims[1]],
    Dim2 = row_coord[, dims[2]], # Assuming you want to plot the first two dimensions
    Label = row_names_ca,
    Type = paste0(legend_1)
  )
  
  df_cols <- data.frame(
    Dim1 = col_coord[, dims[1]],
    Dim2 = col_coord[, dims[2]], # Assuming you want to plot the first two dimensions
    Label = col_names_ca,
    Type = paste0(legend_2)
  )
  
  plot_data <- rbind(df_rows, df_cols)
  
  # Create ggplot
  ca_plot <- ggplot(plot_data, aes(x = Dim1, y = Dim2, color = Type, shape = Type)) +
    geom_point(size = 1) +
    geom_text(aes(label = Label), nudge_y = 0.15, size = 3, show.legend = FALSE) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "grey50") +
    geom_vline(xintercept = 0, linetype = "dashed", color = "grey50") +
    scale_color_manual(values = c("blue", "red")) + # Assign colors based on order
    scale_shape_manual(values = c(19, 19)) + # Assign shapes based on order
    labs(
      x = paste0("Dimensión ",dims[1]," (", inertia_dim1_pct, "%)"),
      y = paste0("Dimensión ",dims[2]," (", inertia_dim2_pct, "%)"),
      title = paste("Análisis de Correspondencia"),
      color = "Tipo",
      shape = "Tipo"
    ) +
    theme_minimal(base_size = 11) +
    theme(
      plot.title = element_text(hjust = 0.5),
      legend.position = "bottom" # Changed to bottom for better visibility if types are long
    )
  
  # Dynamically set legend labels for color and shape
  # The names for scale_color_manual and scale_shape_manual values
  # should match the actual values in plot_data$Type
  type_levels <- unique(plot_data$Type)
  color_values <- setNames(c("blue", "red")[1:length(type_levels)], type_levels)
  shape_values <- setNames(c(19, 19)[1:length(type_levels)], type_levels) # Using same shape for simplicity
  
  ca_plot <- ca_plot + 
    scale_color_manual(values = color_values, name = "Tipo") +
    scale_shape_manual(values = shape_values, name = "Tipo")
  return(list(data = plot_data, plot = ca_plot, ca_results = ca_results))
}

# Read and preprocess data
data = "./data/uscrime.csv"
crime_data <- read.csv(data, header = FALSE, sep = ",", stringsAsFactors = FALSE)
rownames(crime_data) <- crime_data$V1
crime_data <- crime_data[, -1]
region_data <- crime_data[, c(10, 11)]
colnames(crime_data) <- c("land_area", "population_1985", "murder", "rape", 
                           "robbery", "assault", "burglary", "larceny", 
                           "auto_theft", "US_states_region_number", 
                           "US_states_division_number")
crime_data <- crime_data[,c(-1,-2,-10,-11)]

# Plot the results
prueba.1 <- perform_correspondence_analysis(crime_data,
                                legend_1 = "Estados",
                                legend_2 = "Crímenes",
                                dims = c(1, 3),
                                nd=3)


######## Inciso (a) sobre los crimenes ##############
dim3_datos <- prueba.1$data[prueba.1$data$Type == "Crímenes","Dim2"]
dim3_datos <- cbind(rownames(prueba.1$data[prueba.1$data$Type == "Crímenes",]), dim3_datos)
dim3_datos <- as.data.frame(dim3_datos)
dim3_datos <- dim3_datos[!(dim3_datos$V1%in%c("land_area","population_1985")),]

# Convertir la columna 'dim3_datos' a numérica
dim3_datos$dim3_datos <- as.numeric(as.character(dim3_datos$dim3_datos))
# Usamos as.character() primero por si fuera un factor, luego as.numeric()

# Ahora puedes intentar el barplot de nuevo:
colores_barras <- ifelse(dim3_datos$dim3_datos < -4, 
                         "red",   # Color si valor < -4
                         ifelse(dim3_datos$dim3_datos < 0, 
                                "orange", # Color si -4 <= valor < 0
                                "skyblue" # Color si valor >= 0
                         )
)

barplot(dim3_datos$dim3_datos,
        names.arg = dim3_datos$V1,
        col = colores_barras,
        las = 1, # Para que las etiquetas del eje X sean perpendiculares
        xlab = "Crímenes",
        ylab = "Dimensión 3")
grid() # Opcional, para añadir una rejilla


########## Ahora sobre los estados ###########

# Plotting 

prueba.2 <- perform_correspondence_analysis(crime_data,
                                legend_1 = "Estados",
                                legend_2 = "Regiones",
                                dims = c(2, 3),
                                nd=3)
# Extracting the data for plotting
Dim2 <- prueba.2$data[prueba.2$data$Type == "Estados","Dim1"]



dim3_datos <- prueba.1$data[prueba.1$data$Type == "Estados",c("Dim1","Dim2")]
dim3_datos <- cbind(rownames(prueba.1$data[prueba.1$data$Type == "Estados",]),
                    dim3_datos, Dim2, region_data[,1])
dim3_datos <- as.data.frame(dim3_datos)
dim3_datos$Dim1 <- as.numeric(as.character(dim3_datos$Dim1))
dim3_datos$Dim2 <- as.numeric(as.character(dim3_datos$Dim2))

colnames(dim3_datos) <- c("Estado", "Dim1", "Dim3", "Dim2", "Region")

dim3_datos$Region <- factor(dim3_datos$Region)
dim3_datos


library("plotly")
# Crear el gráfico 3D con plotly
fig <- plot_ly(data = dim3_datos, 
               x = ~Dim1, 
               y = ~Dim2, 
               z = ~Dim3, 
               color = ~Region,  # Colorear los puntos según la región
               colors = "viridis", # Puedes elegir otras paletas de colores, ej: "Set1", "Paired"
               type = 'scatter3d', 
               mode = 'markers',
               marker = list(size = 5, opacity = 0.8), # Ajustar tamaño y opacidad de los puntos
               text = ~paste("Estado:", Estado, 
                             "<br>Dim1:", round(Dim1, 2),
                             "<br>Dim2:", round(Dim2, 2),
                             "<br>Dim3:", round(Dim3, 2),
                             "<br>Región:", Region), # Texto que aparece al pasar el cursor
               hoverinfo = 'text') # Mostrar solo el texto personalizado en hover

# Configurar el diseño del gráfico (títulos de ejes, título principal)
fig <- fig %>% layout(title = "Visualización 3D de Estados por Dimensiones del AC",
                      scene = list(xaxis = list(title = 'Dimensión 1'),
                                   yaxis = list(title = 'Dimensión 2'),
                                   zaxis = list(title = 'Dimensión 3'),
                                   aspectratio = list(x=1,y=1,z=1)), # Modo de aspecto para que las dimensiones se vean igual),
                      legend = list(title = list(text = '<b>Región</b>'))) # Título para la leyenda

# Mostrar el gráfico
fig
