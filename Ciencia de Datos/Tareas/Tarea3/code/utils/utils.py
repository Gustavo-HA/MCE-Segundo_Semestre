import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
from pathlib import Path
# Aprendizaje No Supervisado
from sklearn.cluster import (
    AgglomerativeClustering,
    DBSCAN,
    KMeans,
    MeanShift,
    SpectralClustering
)
from sklearn.decomposition import (
    KernelPCA
)
# Métricas
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    confusion_matrix,
    silhouette_score
)
import utils.utils as u
import cv2
import os


def obtener_histograma_gris(imagen):
    """
    Calcula el histograma en escala de grises de una imagen.

    Parámetros:
    - imagen: Imagen en formato OpenCV.

    Retorna:
    - histograma normalizado en un vector 1D.
    """
    gris = cv2.cvtColor(
        imagen, cv2.COLOR_BGR2GRAY)  # Convertir a escala de grises
    hist = cv2.calcHist([gris], [0], None, [256], [
                        0, 256])  # Calcular histograma
    # Normalizar y convertir a vector 1D
    hist = cv2.normalize(hist, hist).flatten()
    return hist


def obtener_histograma_color(imagen):
    """
    Calcula los histogramas de los canales R, G y B de una imagen.

    Parámetros:
    - imagen: Imagen en formato OpenCV.

    Retorna:
    - Vector concatenado de histogramas de los 3 canales.
    """
    hist_color = []
    for i in range(3):  # Canales B, G, R
        hist = cv2.calcHist([imagen], [i], None, [256], [0, 256])  # Histograma
        # Normalizar y convertir a vector 1D
        hist = cv2.normalize(hist, hist).flatten()
        hist_color.extend(hist)  # Concatenar al vector final
    return np.array(hist_color)


def get_vector(ruta_imagen, tipo="gris"):  # tipo = {gris, color, combinado}
    img = cv2.imread(ruta_imagen)  # Leer imagen
    if img is not None:
        # Obtener histogramas
        if tipo == "gris" or tipo == "combinado":
            hist_gris = obtener_histograma_gris(img)
        if tipo == "color" or tipo == "combinado":
            hist_color = obtener_histograma_color(img)

        if tipo == "combinado":
            vector_final = np.concatenate((hist_gris, hist_color))
        elif tipo == "gris":
            vector_final = hist_gris
        elif tipo == "color":
            vector_final = hist_color

    return vector_final
