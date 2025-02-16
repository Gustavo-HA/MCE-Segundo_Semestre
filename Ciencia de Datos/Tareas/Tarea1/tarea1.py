# Preparación: Importar librerías.

from pandas.plotting import parallel_coordinates
import numpy as np
from ucimlrepo import fetch_ucirepo
from time import perf_counter
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import k_means
from sklearn.cluster import AgglomerativeClustering

# Debido al warning
# UserWarning: KMeans is known to have a memory leak on Windows with MKL,
# when there are less chunks than available threads.
# You can avoid it by setting the environment variable OMP_NUM_THREADS=1.
import os

os.environ["OMP_NUM_THREADS"] = "1"


# 1. Implementacion de algoritmos

def distancia_minkowski(x: np.ndarray, y: np.ndarray,
                        p: float):
    """
    Distancia de Minkowski de orden p.

    Parámetros
    ----------
    x : Vector X, mismas dimensiones que Y
    y : Vector Y, mismas dimensiones que X
    p : Orden

    Devuelve
    --------
    distancia : Magnitud distancia entre X a Y de orden p.
    """
    return np.sum(np.abs(x-y)**p)**(1/p)


def jerarquico_aglomerativo(X: np.ndarray, n_clusters: int = 2, umbral: float = None,
                            enlace="single-linkage", p=2) -> np.ndarray:
    """
    Clustering Jerárquico Aglomerativo.
    -----------------------------------

    Calcula los clusters mediante el Clusterin Jerárquico kmedias.  
    Se implementa la distancia de Minkowski de orden p.

    Parámetros
    ----------
    ``X`` : Matriz de datos.
    ``n_clusters`` : Número de clusters. 
        (Originalmente creí que especificabamos el numero de clusters)
    ``umbral`` : Umbral de distancia para detener el algoritmo.
    ``enlace`` : Tipo de enlace. Puede ser "single-linkage", "complete-linkage" o "average-linkage".
    ``p`` : Orden de la distancia de Minkowski.

    Devuelve
    --------
    ``label`` : El label[i] representa el indice del cluster al que pertenece la i-esima observación
    """

    # Primero verificamos que el enlace de input sea correcto
    if enlace not in ["single-linkage", "complete-linkage", "average-linkage"]:
        raise ValueError(f"Argumento 'enlace' puede tener valores ('single-linkage',"
                         f"'complete-linkage', 'average-linkage'), se introdujo el valor '{enlace}'.")

    n = X.shape[0]

    # Cada punto es un cluster
    clusters = [[i] for i in range(n)]

    # Inicializar matriz de disimilitud
    matriz_disim = np.array([[distancia_minkowski(X[i], X[j], p)
                              for j in range(n)]
                             for i in range(n)])

    # Hacer hasta obtener n_clusters clusters
    for t in range(n-n_clusters):

        # Encontrar los clusters más cercanos
        minimo = np.inf
        for i in range(n-t):
            for j in range(n-t):
                if i != j and i < j and matriz_disim[i, j] < minimo:
                    minimo = matriz_disim[i, j]
                    min_i = i
                    min_j = j

        # Detenerse si la distancia menor es mayor al umbral
        if umbral is not None and minimo > umbral:
            break

        clusters[min_i] += clusters[min_j]
        clusters.pop(min_j)

        # Distancias segun enlace
        if enlace == "single-linkage":
            # Distancia mínima del cluster ij al resto de clusters
            distancias_nuevoCluster = np.array([min(matriz_disim[min_i, k], matriz_disim[min_j, k])
                                                for k in range(n-t)
                                                if k != min_i and k != min_j])
            distancias_nuevoCluster = np.insert(
                distancias_nuevoCluster, min_i, 0)
        elif enlace == "complete-linkage":
            # Distancia maxima del cluster ij al resto de clusters
            distancias_nuevoCluster = np.array([max(matriz_disim[min_i, k], matriz_disim[min_j, k])
                                                for k in range(n-t)
                                                if k != min_i and k != min_j])
            distancias_nuevoCluster = np.insert(
                distancias_nuevoCluster, min_i, 0)
        elif enlace == "average-linkage":
            distancias_nuevoCluster = np.zeros(n-t-1)
            n_ij = len(clusters[min_i])
            for cluster_k in clusters[:min_i] + clusters[min_i+1:]:
                n_k = len(cluster_k)
                indice_k = clusters.index(cluster_k)
                for vector_k in cluster_k:
                    for vector_ij in clusters[min_i]:
                        distancias_nuevoCluster[indice_k] += distancia_minkowski(
                            vector_k, vector_ij, p)
                distancias_nuevoCluster[indice_k] /= n_ij * n_k

        # Actualizar matriz de disimilaridad
        matriz_disim = np.delete(matriz_disim, min_j, axis=0)
        matriz_disim = np.delete(matriz_disim, min_j, axis=1)
        matriz_disim[min_i] = distancias_nuevoCluster
        matriz_disim[:, min_i] = distancias_nuevoCluster

    # Devolver etiquetas
    labels = np.full(n, -1)  # Los outliers tendrán indice -1
    for i in range(n_clusters):
        for indice in clusters[i]:
            labels[indice] = i

    return labels


def k_medias(X: np.ndarray, n_clusters: int,
             max_iter=100, epsilon=1e-4, p=2,
             semilla: int | None = None):
    """
    Algoritmo K-means
    -----------------

    Clustering mediante el algoritmo K-means.  
    Se implementa la distancia de Minkowski de orden p.

    Parámetros
    ----------
    ``X`` : Matriz de datos.
    ``n_clusters`` : Número de clusters.
    ``max_iter`` : Número máximo de iteraciones.
    ``epsilon`` : Conergencia del algoritmo.
    ``p`` : Orden de la distancia de Minkowski.

    Devuelve
    --------
    ``centroides`` : ndarray (n_clusters, n_features)

    ``label`` : El label[i] representa el indice del centroide más cercano a la i-esima observación.
    """
    n = X.shape[0]
    labels = np.zeros(n)

    # Se proponen clusters
    if semilla is not None:
        np.random.seed(semilla)
    centroides = X[np.random.choice(n, n_clusters, replace=False)]
    centroides_pasados = None

    convergencia = False
    for i in range(max_iter):

        # Condicion de convergencia
        if centroides_pasados is not None:
            # Calculamos la norma de la diferencia entre los centroides actuales y los anteriores
            diferencias = [distancia_minkowski(centroides[i], centroides_pasados[i], p)
                           for i in range(n_clusters)]
            if max(diferencias) < epsilon:
                convergencia = True
                break

        # Rellenar los clusters
        clusters = [[] for _ in range(n_clusters)]
        for i in range(n):
            c_i = np.argmin([distancia_minkowski(X[i], centroide, p)**p
                             for centroide in centroides])
            clusters[c_i].append(X[i])
            labels[i] = c_i

        # Guardar centroides actuales
        centroides_pasados = centroides.copy()

        # Actualizar centroides como la media de sus puntos
        for j in range(n_clusters):

            # Si está vacío el cluster, no se actualiza
            if not clusters[j]:
                continue

            centroides[j] = np.mean(clusters[j], axis=0)

    if not convergencia:
        print(f"No convergió en {max_iter} iteraciones.")

    return (centroides, labels)


def coeficiente_silueta(X: np.ndarray, labels: list) -> float:
    """
    Silueta promedio de un clustering.
    -----------------------------------

    Calcula el coeficiente de silueta promedio de un clustering.  
    Se implementa la distancia de Minkowski de orden p.

    Parámetros
    ----------
    ``X`` : Matriz de datos.
    ``label`` : El label[i] representa el indice del centroide más cercano a la i-esima observación.

    Devuelve
    --------
    ``s_k_promedio``: Coeficiente de silueta promedio.
    """
    # Numero de clusters y de puntos
    labels_unicos = set(labels)
    n_clusters = len(labels_unicos)
    n = X.shape[0]

    clusters = [[X[i] for i in range(n) if label == labels[i]]
                for label in labels_unicos]

    # Inicializar s_k en ceros
    s_k = np.zeros(n_clusters)

    # Iteramos sobre los clusters
    for k in range(n_clusters):
        vectores_k = clusters[k]

        # Dejamos s_k = 0 para cuando
        # el cluster está vacío o tiene 1 elemento.
        # https://en.wikipedia.org/wiki/Silhouette_(clustering)
        if len(vectores_k) == 0 or len(vectores_k) == 1:
            continue

        # Iteramos sobre cada vector del cluster
        for vector in vectores_k:
            # Calcular a_i
            a_i = np.mean([distancia_minkowski(vector, vector_k, 2)
                           for vector_k in vectores_k
                           if not np.array_equal(vector, vector_k)])

            # Calcular b_ic (tamaño k-1) y obtener b_i
            b_ic = [np.mean([distancia_minkowski(vector, vector_c, 2)
                             for vector_c in clusters[c]])
                    for c in range(n_clusters) if c != k]
            b_i = min(b_ic)

            s_i = (b_i - a_i) / max(a_i, b_i)

            # acumulamos s_i de cada cluster
            s_k[k] += s_i

    # Regresamos el promedio de todas las s_i
    s_k_promedio = s_k.sum() / n
    return s_k_promedio


# 2. Evaluación de Resultados.

# 2.1 la base de datos
"""
Este conjunto de datos es sobre tres especies diferentes de la flor Iris, 
las cuales se definen en términos de las dimensiones de sus sépalos y pétalos.

El objetivo es realizar un análisis sobre las distintas especies de Iris 
utilizando las herramientas implementadas anteriormente, junto con métodos de visualización.

1. *Número de id*: 1 a 150  
2. *Sepal.Length*: Longitud del sépalo (cm)  
3. *Sepal.Width*: Ancho del sépalo (cm)  
4. *Petal.Length*: Longitud del pétalo (cm)  
5. *Petal.Width*: Ancho del pétalo (cm)  
6. *Species*: Especie de la flor (atributo de clase):  
    - setosa  
    - versicolor  
    - virginica 
"""

# Se importa directamente mediante la librería de UCI ML.

# fetch dataset
iris = fetch_ucirepo(id=53)

# data (as pandas dataframes)
X = pd.DataFrame(iris.data.features)
y = pd.DataFrame(iris.data.targets)

df = pd.concat([X, y], axis=1)


# Para input en los algoritmos
X_numpy = X.to_numpy()

# Renombramos las clases con 0, 1 y 2, todo esto para input
# de la función coeficiente_silueta.
y["class"] = y["class"].apply(lambda clase: 0
                              if clase == "Iris-setosa" else 1
                              if clase == "Iris-versicolor" else 2)
y_label = list(y["class"].to_numpy())


# 2.2. Describir la base de datos y sus comportamientos

scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(
    df[df.columns[:-1]]), columns=df.columns[:-1])
df_scaled = pd.concat([df_scaled, df["class"]], axis=1)

# Después ocuparemos el X escalado
X_scaled_numpy = df_scaled[df_scaled.columns[:-1]].to_numpy()

# pairplot
sns.set_theme("paper")

plt.figure(figsize=(10, 8))
sns.pairplot(df, hue="class", palette="Set1", height=1.5, diag_kind="kde")

plt.plot()

plt.figure(figsize=(10, 8))
sns.pairplot(df_scaled, hue="class", palette="Set1",
             height=1.5, diag_kind="kde")
plt.suptitle("Datos estandarizados", y=1.02)

plt.plot()

# boxplots
f, ax = plt.subplots(1, 1, figsize=(6, 4))
f.suptitle("Class - Petal Width", size=18)

sns.boxplot(data=df, x="class", y="petal width", ax=(ax), hue="class")
ax.set_xlabel("Class", size=14, alpha=0.8)
ax.set_ylabel("Petal Width [cm]", size=14, alpha=0.8)


# Determinar porcentaje de obs versicolor sobre virginica.
min_virginica = df[df["class"] == "Iris-virginica"]["petal width"].min()
n_versi_sobre_virginica = df[df["class"] == "Iris-versicolor"]["petal width"].apply(lambda valor: 1
                                                                                    if valor > min_virginica else 0).sum()
porcentaje_vers_sobre_virg = n_versi_sobre_virginica / \
    df[df["class"] == "Iris-versicolor"].count(axis=0).iloc[0]
print(
    f"Porcentaje de traslape: Versicolor sobre Virginica {porcentaje_vers_sobre_virg:.2%}")

f, ax = plt.subplots(1, 1, figsize=(6, 4))
f.suptitle("Class - Petal Width [Estandarizado]", size=18)

sns.boxplot(data=df_scaled, x="class", y="petal width", hue="class")
ax.set_xlabel("Class", size=14, alpha=0.8)
ax.set_ylabel("Petal Width", size=14, alpha=0.8)

plt.plot()


# parallel coordinates

fig, ax = plt.subplots(1, 2, figsize=(12, 5))

g1 = parallel_coordinates(frame=df, class_column="class", ax=ax[0])
g2 = parallel_coordinates(df_scaled, class_column="class", ax=ax[1])

g1.set_title("Datos sin estandarizar")
g2.set_title("Datos estandarizados")

plt.plot()


# KDE en 2 dimensiones

fig, ax = plt.subplots(1, 2, figsize=(12, 5))

g1 = sns.kdeplot(df, x="petal width", y="petal length",
                 hue="class", fill=True, ax=ax[0])
g2 = sns.kdeplot(df, x="petal width", y="petal length",
                 hue="class", fill=False, ax=ax[1])

sns.move_legend(g1, loc="upper left")
sns.move_legend(g2, loc="upper left")

fig, ax = plt.subplots(1, 2, figsize=(12, 5))
plt.suptitle("Datos estandarizados")
g1 = sns.kdeplot(df_scaled, x="petal width", y="petal length",
                 hue="class", fill=True, ax=ax[0])
g2 = sns.kdeplot(df_scaled, x="petal width", y="petal length",
                 hue="class", fill=False, ax=ax[1])

sns.move_legend(g1, loc="upper left")
sns.move_legend(g2, loc="upper left")

plt.plot()

# Matriz de correlacion


plt.figure(figsize=(6, 4))
columnas = df.columns[:-1]  # Todas menos la clase
corr_matrix = df[columnas].corr()
g1 = sns.heatmap(corr_matrix, cmap="coolwarm", fmt=".2f", annot=True)
plt.plot()


plt.figure(figsize=(6, 4))
plt.suptitle("Datos estandarizados")
columnas = df_scaled.columns[:-1]  # Todas menos la clase
corr_matrix = df_scaled[columnas].corr()
g1 = sns.heatmap(corr_matrix, cmap="coolwarm", fmt=".2f", annot=True)
plt.plot()

# 2.3 Experimentar con algoritmos


# Para CJA ver clusters.
def comparar_aglomerativo_kdeplot(enlace, umbral=None, p=2):
    labels_aglomerativo = jerarquico_aglomerativo(
        X_numpy, umbral=umbral, enlace=enlace, p=p)
    labels_aglomerativo_scaled = jerarquico_aglomerativo(
        X_scaled_numpy, umbral=umbral, enlace=enlace, p=p)

    df_aglomerativo = df.copy()
    df_aglomerativo["class aglomerativo"] = np.array(labels_aglomerativo)
    df_aglomerativo["class aglomerativo scaled"] = np.array(
        labels_aglomerativo_scaled)
    fig, ax = plt.subplots(1, 3, figsize=(12, 5))
    g1 = sns.kdeplot(df_aglomerativo, x="petal width", y="petal length",
                     hue="class", fill=False, ax=ax[0], palette="Set1")
    g2 = sns.kdeplot(df_aglomerativo, x="petal width", y="petal length",
                     hue="class aglomerativo", fill=False, ax=ax[1], palette="Set1", warn_singular=False)
    g3 = sns.kdeplot(df_aglomerativo, x="petal width", y="petal length",
                     hue="class aglomerativo scaled", fill=False, ax=ax[2], palette="Set1", warn_singular=False)

    # Titulos
    # Cada uno imprime su coeficiente de silueta.
    X = df_aglomerativo[df_aglomerativo.columns[:-3]].to_numpy()
    cs_1 = coeficiente_silueta(X, df_aglomerativo["class"].to_list())
    g1.set_title(f"Original: CS = {cs_1:.3f}")
    cs_2 = coeficiente_silueta(
        X, df_aglomerativo["class aglomerativo"].to_list())
    g2.set_title(f"No estandarizado: CS = {cs_2:.3f}")
    cs_3 = coeficiente_silueta(
        X, df_aglomerativo["class aglomerativo scaled"].to_list())
    g3.set_title(f"Estandarizado: CS = {cs_3:.3f}")

    # Leyendas
    # Quiero poner que también me diga cuántos elementos hay en cada cluster
    agrupa_1 = df_aglomerativo.groupby("class aglomerativo").size()
    n_puntos_1 = agrupa_1.to_list()
    etiquetas_1 = agrupa_1.index.to_list()
    labels_1 = [
        f"{etiquetas_1[i]} ({n_puntos_1[i]})" for i in range(len(etiquetas_1))]

    agrupa_2 = df_aglomerativo.groupby("class aglomerativo scaled").size()
    n_puntos_2 = agrupa_2.to_list()
    etiquetas_2 = agrupa_2.index.to_list()
    labels_2 = [
        f"{etiquetas_2[i]} ({n_puntos_2[i]})" for i in range(len(etiquetas_2))]

    sns.move_legend(g1, loc="upper left")
    sns.move_legend(g2, loc="upper left", title="class", labels=labels_1)
    sns.move_legend(g3, loc="upper left", title="class", labels=labels_2)

    # Titulo grande
    plt.suptitle(
        f"Comparación CJA: enlace '{enlace}', umbral = {umbral}, orden = {p}")

    plt.show()


# Con CJA
umbrales = [0.1, 1, 4]
enlaces = ["complete-linkage", "single-linkage"]  # , "average-linkage"]
ordenes = [0.5, 1, 2, 4]


for enlace in enlaces:
    for umbral in umbrales:
        for p in ordenes:
            comparar_aglomerativo_kdeplot(enlace, umbral, p)

# complete-linkage: umbral = 4, p = 2
# single-linkage: umbral = 4, p = 4
# average-linkage: umbral = 4, p = 2

# Con K-means
k_clusters = list(range(2, 10))
ordenes = [0.5, 1, 2, 4]
semilla = 42

lista_orden = list()
lista_k = list()
lista_cs = list()
lista_cs_scaled = list()

for p in ordenes:
    for k in k_clusters:
        lista_orden.append(p)
        lista_k.append(k)
        lista_cs.append(coeficiente_silueta(
            X_numpy, k_medias(X_numpy, k, p=p, semilla=semilla)[1]))
        lista_cs_scaled.append(coeficiente_silueta(
            X_scaled_numpy, k_medias(X_scaled_numpy, k, p=p, semilla=semilla)[1]))

df_cs = pd.DataFrame(
    {"cs": lista_cs, "cs_scaled": lista_cs_scaled, "k": lista_k, "p": lista_orden})


sns.set_theme("paper")
sns.set_style("darkgrid")
f, ax = plt.subplots(1, 2, figsize=(12, 4))
g1 = sns.lineplot(df_cs, x="k", y="cs", hue="p", ax=ax[0])
g2 = sns.lineplot(df_cs, x="k", y="cs_scaled", hue="p", ax=ax[1])
g1.set_title("Datos sin estandarizar", fontdict={'fontsize': 12})
g2.set_title("Datos estandarizados", fontdict={'fontsize': 12})

sns.move_legend(g1, loc="upper right", title="Orden DM")
sns.move_legend(g2, loc="upper right", title="Orden DM")

ax[0].set_ylabel("CS : Coeficiente de Siluetas")
ax[0].set_xlabel("K : Número de clústers")

ax[1].set_ylabel("CS : Coeficiente de Siluetas")
ax[1].set_xlabel("K : Número de clústers")

g1.set_xticks(k_clusters)
g2.set_xticks(k_clusters)
plt.show()


# Para las KDE - ver clusters
def comparar_kmedias_kdeplot(k, p=2):
    labels_aglomerativo = k_medias(X_numpy, n_clusters=k, p=p)[1]
    labels_aglomerativo_scaled = k_medias(X_scaled_numpy, n_clusters=k, p=p)[1]

    df_aglomerativo = df.copy()
    df_aglomerativo["class aglomerativo"] = np.array(labels_aglomerativo)
    df_aglomerativo["class aglomerativo scaled"] = np.array(
        labels_aglomerativo_scaled)
    fig, ax = plt.subplots(1, 3, figsize=(12, 5))
    g1 = sns.kdeplot(df_aglomerativo, x="petal width", y="petal length",
                     hue="class", fill=False, ax=ax[0], palette="Set1")
    g2 = sns.kdeplot(df_aglomerativo, x="petal width", y="petal length",
                     hue="class aglomerativo", fill=False, ax=ax[1], palette="Set1", warn_singular=False)
    g3 = sns.kdeplot(df_aglomerativo, x="petal width", y="petal length",
                     hue="class aglomerativo scaled", fill=False, ax=ax[2], palette="Set1", warn_singular=False)

    # Titulos
    # Cada uno imprime su coeficiente de silueta.
    X = df_aglomerativo[df_aglomerativo.columns[:-3]].to_numpy()
    cs_1 = coeficiente_silueta(X, df_aglomerativo["class"].to_list())
    g1.set_title(f"Original: CS = {cs_1:.3f}")
    cs_2 = coeficiente_silueta(
        X, df_aglomerativo["class aglomerativo"].to_list())
    g2.set_title(f"No estandarizado: CS = {cs_2:.3f}")
    cs_3 = coeficiente_silueta(
        X, df_aglomerativo["class aglomerativo scaled"].to_list())
    g3.set_title(f"Estandarizado: CS = {cs_3:.3f}")

    # Leyendas
    # Quiero poner que también me diga cuántos elementos hay en cada cluster
    agrupa_1 = df_aglomerativo.groupby("class aglomerativo").size()
    n_puntos_1 = agrupa_1.to_list()
    etiquetas_1 = agrupa_1.index.to_list()
    labels_1 = [
        f"{etiquetas_1[i]} ({n_puntos_1[i]})" for i in range(len(etiquetas_1))]

    agrupa_2 = df_aglomerativo.groupby("class aglomerativo scaled").size()
    n_puntos_2 = agrupa_2.to_list()
    etiquetas_2 = agrupa_2.index.to_list()
    labels_2 = [
        f"{etiquetas_2[i]} ({n_puntos_2[i]})" for i in range(len(etiquetas_2))]

    sns.move_legend(g1, loc="upper left")
    sns.move_legend(g2, loc="upper left", title="class", labels=labels_1)
    sns.move_legend(g3, loc="upper left", title="class", labels=labels_2)

    # Titulo grande
    plt.suptitle(f"Comparación K-medias: k = {k} , orden = {p}")

    plt.show()


# No importa mucho el orden de la DM
for orden in ordenes:
    comparar_kmedias_kdeplot(3, orden)


# 3. Visualización (PCA)

# Es sabido que PCA funciona mejor con los datos estandarizados
pca_matrix = PCA(n_components=2).fit_transform(X_scaled_numpy)

# Utilizando el óptimo de k_means
centroides, labels = k_medias(X_numpy, 3, semilla=42)
labels = [int(label) for label in labels]
centroides_pca = PCA(n_components=2).fit_transform(centroides)


df_pca = pd.DataFrame(data=pca_matrix, columns=['PCA1', 'PCA2'])
df_pca['class'] = df["class"]
df_pca['class kmeans'] = labels

f, ax = plt.subplots(1, 2, figsize=(12, 6))
g1 = sns.scatterplot(data=df_pca, x='PCA1', y='PCA2',
                     hue='class', s=75, ax=ax[0])
g2 = sns.scatterplot(data=df_pca, x='PCA1', y='PCA2',
                     hue='class kmeans', s=75, palette="Set1", ax=ax[1])

ax[0].set_xlabel("PCA1")
ax[0].set_ylabel("PCA2")
ax[1].set_xlabel("PCA1")
ax[1].set_ylabel("PCA2")
ax[1].scatter(centroides_pca[:, 0], centroides_pca[:, 1],
              c='black', marker='X', s=100)

sns.move_legend(g1, loc="lower right", title="Class")
sns.move_legend(g2, loc="lower right",  title="Class")

plt.suptitle('Proyección PCA 2D')
plt.tight_layout()
plt.show()


# 4. COMPARACIÓN CON SCIKIT-LEARN

# CJA
inicio = perf_counter()
for _ in range(10):
    jerarquico_aglomerativo(X_numpy)
final = perf_counter()

d1 = final-inicio

print(f"Implementado: Tiempo de ejecución promedio = {d1/10:.4f} segundos")

inicio = perf_counter()
for _ in range(10):
    AgglomerativeClustering().fit(X_numpy)
final = perf_counter()

d2 = final-inicio

print(f"SKLearn: Tiempo de ejecución promedio = {d2/10:.4f} segundos")

print(f"SpeedUp de SKLearn contra Implementación: {d1/d2}")


# KMEANS

inicio = perf_counter()
for _ in range(10):
    k_medias(X_numpy, 3)
final = perf_counter()

d1 = final-inicio

print(f"Implementado: Tiempo de ejecución promedio = {d1/10:.4f} segundos")

inicio = perf_counter()
for _ in range(10):
    k_means(X_numpy, 3)
final = perf_counter()

d2 = final-inicio

print(f"SKLearn: Tiempo de ejecución promedio = {d2/10:.4f} segundos")

print(f"SpeedUp de SKLearn contra Implementación: {d1/d2}")
