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

library("dplyr")

# Read and preprocess data
data = "./data/uscrime.csv"
crime_data <- read.csv(data, header = FALSE, sep = ",", stringsAsFactors = FALSE)
crime_data <- crime_data[, c(-1,-12)]


summed_v2_dplyr <- crime_data %>%
  group_by(V11) %>%
  summarise(Total_V2 = sum(V2, na.rm = TRUE)) # Example for summing V2

# the same for v3 - v10
summed_v3_dplyr <- crime_data %>%
  group_by(V11) %>%
  summarise(Total_V3 = sum(V3, na.rm = TRUE))
summed_v4_dplyr <- crime_data %>%
  group_by(V11) %>%
  summarise(Total_V4 = sum(V4, na.rm = TRUE))
summed_v5_dplyr <- crime_data %>%
  group_by(V11) %>%
  summarise(Total_V5 = sum(V5, na.rm = TRUE))
summed_v6_dplyr <- crime_data %>%
  group_by(V11) %>%
  summarise(Total_V6 = sum(V6, na.rm = TRUE))
summed_v7_dplyr <- crime_data %>%
  group_by(V11) %>%
  summarise(Total_V7 = sum(V7, na.rm = TRUE))
summed_v8_dplyr <- crime_data %>%
  group_by(V11) %>%
  summarise(Total_V8 = sum(V8, na.rm = TRUE))
summed_v9_dplyr <- crime_data %>%
  group_by(V11) %>%
  summarise(Total_V9 = sum(V9, na.rm = TRUE))
summed_v10_dplyr <- crime_data %>%
  group_by(V11) %>%
  summarise(Total_V10 = sum(V10, na.rm = TRUE))

# Combine the summed data into a contingency table
contingency_table <- data.frame(
  V11 = summed_v2_dplyr$V11,
  V2 = summed_v2_dplyr$Total_V2,
  V3 = summed_v3_dplyr$Total_V3,
  V4 = summed_v4_dplyr$Total_V4,
  V5 = summed_v5_dplyr$Total_V5,
  V6 = summed_v6_dplyr$Total_V6,
  V7 = summed_v7_dplyr$Total_V7,
  V8 = summed_v8_dplyr$Total_V8,
  V9 = summed_v9_dplyr$Total_V9,
  V10 = summed_v10_dplyr$Total_V10
)

contingency_table <- contingency_table[,c(-1,-2,-3)]
colnames(contingency_table) <- c("murder", "rape", 
                                "robbery", "assault", "burglary", "larceny", 
                                "auto_theft")

rownames(contingency_table) <- c("1", "2", "3", "4")

anacorr1_3 <- perform_correspondence_analysis(contingency_table, legend_1 = "Regiones",
                                 legend_2 = "Crímenes", dims = c(1,3) , nd = 3)
anacorr2_3 <- perform_correspondence_analysis(contingency_table, legend_1 = "Regiones",
                                 legend_2 = "Crímenes", dims = c(2,3) , nd = 3)

anacorr2_3$plot
anacorr1_3$plot

data <- cbind(anacorr1_3$data, anacorr2_3$data[,c("Dim1")])
colnames(data)[2] <- "Dim3"
colnames(data)[5] <- "Dim2"

label <- c("1", "2", "3", "4", "murder", "rape", 
           "robbery", "assault", "burglary", "larceny", 
           "auto_theft")
data <- cbind(data, Label = label)

# 3d plot with plotly coloring by crime/region
library(plotly)
fig <- plot_ly(data = data, x = ~Dim1, y = ~Dim2, z = ~Dim3, color = ~Type, colors = c("blue", "red")) %>%
  add_markers(text = ~Label, size = 5) %>%
  layout(scene = list(xaxis = list(title = 'Dim1'),
                      yaxis = list(title = 'Dim2'),
                      zaxis = list(title = 'Dim3')),
         title = "Análisis de Correspondencia en 3D")
fig <- fig %>% layout(title = "Visualización 3D del AC",
                      scene = list(xaxis = list(title = 'Dimensión 1'),
                                   yaxis = list(title = 'Dimensión 2'),
                                   zaxis = list(title = 'Dimensión 3'),
                                   aspectratio = list(x=1,y=1,z=1)))

fig


