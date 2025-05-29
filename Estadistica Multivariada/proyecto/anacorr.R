library("ca")
library("ggplot2")

data <- read.csv("./data/bank-full.csv", sep = ";")

perform_correspondence_analysis <- function(input_data, col1_name, col2_name,
                                            legend_1, legend_2) {
  # Select the specified columns
  selected_data <- input_data[, c(col1_name, col2_name)]
  
  # Create a contingency table
  contingency_table <- table(selected_data)
  
  # Perform Chi-squared test (optional, but good for checking association)
  print(chisq.test(contingency_table))
  
  # Perform Correspondence Analysis
  ca_results <- ca(contingency_table)
  
  # Extract row and column coordinates
  row_coord <- ca_results$rowcoord
  col_coord <- ca_results$colcoord
  
  row_names_ca <- rownames(row_coord)
  col_names_ca <- rownames(col_coord)
  
  # Calculate inertia (variance explained) for each dimension
  inertia <- ca_results$sv^2 / sum(ca_results$sv^2)
  inertia_dim1_pct <- round(inertia[1] * 100, 1)
  inertia_dim2_pct <- round(inertia[2] * 100, 1)
  
  # Cumulative inertia (optional, good for deciding number of dimensions)
  print(cumsum(ca_results$sv^2) / sum(ca_results$sv^2))
  
  # Prepare data for plotting
  df_rows <- data.frame(
    Dim1 = row_coord[, 1],
    Dim2 = row_coord[, 2], # Assuming you want to plot the first two dimensions
    Label = row_names_ca,
    Type = paste0(legend_1)
  )
  
  df_cols <- data.frame(
    Dim1 = col_coord[, 1],
    Dim2 = col_coord[, 2], # Assuming you want to plot the first two dimensions
    Label = col_names_ca,
    Type = paste0(legend_2)
  )
  
  plot_data <- rbind(df_rows, df_cols)
  
  # Create ggplot
  ca_plot <- ggplot(plot_data, aes(x = Dim1, y = Dim2, color = Type, shape = Type)) +
    geom_point(size = 4) +
    geom_text(aes(label = Label), nudge_y = 0.15, size = 3, show.legend = FALSE) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "grey50") +
    geom_vline(xintercept = 0, linetype = "dashed", color = "grey50") +
    scale_color_manual(values = c("blue", "red")) + # Assign colors based on order
    scale_shape_manual(values = c(19, 19)) + # Assign shapes based on order
    labs(
      x = paste0("Dimensión 1 (", inertia_dim1_pct, "%)"),
      y = paste0("Dimensión 2 (", inertia_dim2_pct, "%)"),
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
  
  return(ca_plot)
}


perform_correspondence_analysis(
  data[data$job!="unknown",], 
  "marital", 
  "job",
  "Estado Civil",
  "Tipo de Trabajo")


perform_correspondence_analysis(
  data[data$education!="unknown" & data$job != "unknown",],
  "job",
  "education",
  "Tipo de Trabajo",
  "Nivel de Educación")
