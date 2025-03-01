import numpy as np
import heapq
from typing import Literal


def dijkstra(grafo: dict[str, int], inicio: str):
    """
    Implementa el algoritmo de Dijkstra para encontrar la ruta más corta desde un nodo de inicio 
    hasta todos los demás nodos en un grafo ponderado representado como un diccionario de adyacencias.

    Parámetros:
    ----------
    grafo : dict   \\
        Un diccionario donde las claves son nodos y los valores son diccionarios con los nodos vecinos 
        y sus respectivos pesos. Ejemplo:
        {
            'A': {'B': 1, 'C': 4},
            'B': {'A': 1, 'C': 2, 'D': 5},
            'C': {'A': 4, 'B': 2, 'D': 1},
            'D': {'B': 5, 'C': 1}
        }
    inicio : hashable
        Nodo desde el cual se calcularán las distancias mínimas a los demás nodos.

    Retorna:
    -------
    dict
        Un diccionario donde las claves son los nodos y los valores son la distancia mínima desde 
        el nodo de inicio hasta ese nodo.

    Complejidad:
    -----------
    O((V + E) log V), donde V es el número de nodos y E es el número de aristas.
    """
    # Inicializar distancias a todos los nodos como infinito y el nodo de inicio como 0
    distancias = {nodo: float('inf') for nodo in grafo}
    distancias[inicio] = 0

    # Inicializar una cola de prioridad (heap) con el nodo de inicio
    cola_prioridad = [(0, inicio)]

    while cola_prioridad:
        # Obtener el nodo con la distancia más corta de la cola de prioridad
        distancia_actual, nodo_actual = heapq.heappop(cola_prioridad)

        # Si encontramos una distancia más corta a través de este nodo, actualizamos la distancia
        if distancia_actual > distancias[nodo_actual]:
            continue

        # Explorar los vecinos del nodo actual
        for vecino, peso in grafo[nodo_actual].items():
            distancia = distancia_actual + peso

            # Si encontramos una distancia más corta hacia el vecino, actualizamos la distancia
            if distancia < distancias[vecino]:
                distancias[vecino] = distancia
                heapq.heappush(cola_prioridad, (distancia, vecino))

    return distancias


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
                            enlace: Literal["single-linkage", "complete-linkage",
                                            "average-linkage"] = "single-linkage",
                            p=2) -> np.ndarray:
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
