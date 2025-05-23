# Problema 4
library("smacof")

# Distancias entre ciudades
data <- read.csv("distancias ciudades.csv")
data <- data[,-1]
data

mds_1 <- smacofSym(data, ndim = 1, type ="ratio", itmax = 200, eps = 1e-06)
mds_2 <- smacofSym(data, ndim = 2, type ="ratio", itmax = 200, eps = 1e-06)
mds_3 <- smacofSym(data, ndim = 3, type ="ratio", itmax = 200, eps = 1e-06)


stresses <- c(mds_1$stress, mds_2$stress, mds_3$stress)

plot(stresses, type = "b", xlab = "Dimensiones", ylab = "Stress", main = "Stress vs Dimensiones")
# 2 es la dimensionalidad adecuada


mds_2_random <- smacofSym(data, ndim = 2, type ="ratio", itmax = 200, eps = 1e-06,
                          init="random")


# Plot de ciudades no random
plot(mds_2$conf, pch = 19, col = "blue", xlab = "Dim1", ylab = "Dim2",
     main = "MDS 2D - Ciudades")
text(mds_2$conf, labels = rownames(mds_2$conf), cex = 0.7, pos = 3)



# Plot de ciudades random
plot(mds_2_random$conf, pch = 19, col = "red", xlab = "Dim1", ylab = "Dim2",
     main = "MDS 2D - Ciudades (Random)")
text(mds_2_random$conf, labels = rownames(mds_2_random$conf), cex = 0.7, pos = 3)

# Comparamos stresses
stresses[2]
mds_2_random$stress
