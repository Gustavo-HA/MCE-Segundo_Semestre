# Optimización - Evolución Diferencial
# Gustavo 

import numpy as np
import time

# Función de Ackley
def funcion_ackley(vector_solucion, num_dimensiones):
    param_a = 20
    param_b = 0.2
    param_c = 2 * np.pi
    termino_suma_cuadrados = -param_b * np.sqrt((1/num_dimensiones) * np.sum(vector_solucion**2))
    termino_suma_cosenos = (1/num_dimensiones) * np.sum(np.cos(param_c * vector_solucion))
    resultado = param_a + np.exp(1) - param_a * np.exp(termino_suma_cuadrados) - np.exp(termino_suma_cosenos)
    return resultado

# Random, cruza binomial
def ed_aleatorio_1_bin(funcion_objetivo, limites_busqueda, 
                       num_dimensiones, tamano_poblacion, 
                       factor_F, tasa_CR, max_generaciones, 
                       tolerancia_convergencia=1e-7, paciencia_convergencia=25, 
                       semilla_aleatoria=None):
    estado_aleatorio = np.random.RandomState(semilla_aleatoria)
    limite_inferior, limite_superior = np.asarray(limites_busqueda).T
    
    # Inicializar población (normalizada en [0,1])
    poblacion_actual = estado_aleatorio.rand(tamano_poblacion, num_dimensiones)
    
    # Escalar población a los límites reales para la evaluación inicial
    poblacion_escalada = limite_inferior + poblacion_actual * (limite_superior - limite_inferior)
    aptitud_poblacion = np.asarray([funcion_objetivo(individuo, num_dimensiones) for individuo in poblacion_escalada])

    mejor_aptitud_global = np.min(aptitud_poblacion)
    mejor_solucion_global = poblacion_escalada[np.argmin(aptitud_poblacion)].copy()
    
    generaciones_ejecutadas = max_generaciones
    historial_aptitud = []

    for gen_actual in range(max_generaciones):
        vectores_de_prueba = np.zeros_like(poblacion_actual)
        
        for i in range(tamano_poblacion):
            indices_candidatos = [idx for idx in range(tamano_poblacion) if idx != i]
            idx_r1, idx_r2, idx_r3 = estado_aleatorio.choice(indices_candidatos, 3, replace=False)
            
            vector_mutante = poblacion_actual[idx_r1] + factor_F * (poblacion_actual[idx_r2] - poblacion_actual[idx_r3])
            vector_mutante = np.clip(vector_mutante, 0, 1) # Recortar en espacio normalizado
            
            puntos_de_cruce = estado_aleatorio.rand(num_dimensiones) < tasa_CR
            if not np.any(puntos_de_cruce): # Asegurar al menos un punto de cruce
                puntos_de_cruce[estado_aleatorio.randint(0, num_dimensiones)] = True
            
            vector_de_prueba_actual = np.where(puntos_de_cruce, vector_mutante, poblacion_actual[i])
            vectores_de_prueba[i] = vector_de_prueba_actual

        # Escalar vectores de prueba
        vectores_de_prueba_escalados = limite_inferior + vectores_de_prueba * (limite_superior - limite_inferior)
        aptitud_vectores_de_prueba = np.asarray([funcion_objetivo(individuo, num_dimensiones) for individuo in vectores_de_prueba_escalados])
        
        # Selección
        mascara_mejora = aptitud_vectores_de_prueba < aptitud_poblacion
        poblacion_actual[mascara_mejora] = vectores_de_prueba[mascara_mejora]
        aptitud_poblacion[mascara_mejora] = aptitud_vectores_de_prueba[mascara_mejora]

        aptitud_mejor_gen_actual = np.min(aptitud_poblacion)
        if aptitud_mejor_gen_actual < mejor_aptitud_global:
            mejor_aptitud_global = aptitud_mejor_gen_actual
            mejor_solucion_global = limite_inferior + poblacion_actual[np.argmin(aptitud_poblacion)] * (limite_superior - limite_inferior)

        historial_aptitud.append(mejor_aptitud_global)
        if gen_actual >= paciencia_convergencia:
            if abs(historial_aptitud[gen_actual] - historial_aptitud[gen_actual-paciencia_convergencia]) < tolerancia_convergencia:
                generaciones_ejecutadas = gen_actual + 1
                break
                
    solucion_final_escalada = limite_inferior + poblacion_actual[np.argmin(aptitud_poblacion)] * (limite_superior - limite_inferior)
    aptitud_final = np.min(aptitud_poblacion)
    
    return solucion_final_escalada, aptitud_final, generaciones_ejecutadas

# Best, cruza binomial
def ed_mejor_1_bin(funcion_objetivo, limites_busqueda, num_dimensiones, 
                   tamano_poblacion, factor_F, tasa_CR, max_generaciones, 
                   tolerancia_convergencia=1e-7, paciencia_convergencia=25, 
                   semilla_aleatoria=None):
    estado_aleatorio = np.random.RandomState(semilla_aleatoria)
    limite_inferior, limite_superior = np.asarray(limites_busqueda).T

    poblacion_actual = estado_aleatorio.rand(tamano_poblacion, num_dimensiones) # Normalizada en [0,1]
    
    poblacion_escalada = limite_inferior + poblacion_actual * (limite_superior - limite_inferior)
    aptitud_poblacion = np.asarray([funcion_objetivo(individuo, num_dimensiones) for individuo in poblacion_escalada])
    
    mejor_aptitud_global = np.min(aptitud_poblacion)
    mejor_solucion_global = poblacion_escalada[np.argmin(aptitud_poblacion)].copy()
    
    generaciones_ejecutadas = max_generaciones
    historial_aptitud = []

    for gen_actual in range(max_generaciones):
        idx_mejor_pob_actual = np.argmin(aptitud_poblacion)
        vector_mejor_normalizado = poblacion_actual[idx_mejor_pob_actual] # Mejor en espacio normalizado
        
        vectores_de_prueba = np.zeros_like(poblacion_actual)

        for i in range(tamano_poblacion):
            indices_candidatos = [idx for idx in range(tamano_poblacion) if idx != i and idx != idx_mejor_pob_actual]
            if len(indices_candidatos) < 2: 
                 todos_los_demas_indices = [idx for idx in range(tamano_poblacion) if idx != i]
                 if not todos_los_demas_indices: 
                     idx_r2, idx_r3 = i, i
                 elif len(todos_los_demas_indices) < 2:
                     idx_r2, idx_r3 = todos_los_demas_indices[0], todos_los_demas_indices[0]
                 else:
                     idx_r2, idx_r3 = estado_aleatorio.choice(todos_los_demas_indices, 2, replace=False)
            else:
                 idx_r2, idx_r3 = estado_aleatorio.choice(indices_candidatos, 2, replace=False)
            
            vector_mutante = vector_mejor_normalizado + factor_F * (poblacion_actual[idx_r2] - poblacion_actual[idx_r3])
            vector_mutante = np.clip(vector_mutante, 0, 1)
            
            puntos_de_cruce = estado_aleatorio.rand(num_dimensiones) < tasa_CR
            if not np.any(puntos_de_cruce):
                puntos_de_cruce[estado_aleatorio.randint(0, num_dimensiones)] = True
            
            vector_de_prueba_actual = np.where(puntos_de_cruce, vector_mutante, poblacion_actual[i])
            vectores_de_prueba[i] = vector_de_prueba_actual

        vectores_de_prueba_escalados = limite_inferior + vectores_de_prueba * (limite_superior - limite_inferior)
        aptitud_vectores_de_prueba = np.asarray([funcion_objetivo(individuo, num_dimensiones) for individuo in vectores_de_prueba_escalados])
        
        mascara_mejora = aptitud_vectores_de_prueba < aptitud_poblacion
        poblacion_actual[mascara_mejora] = vectores_de_prueba[mascara_mejora]
        aptitud_poblacion[mascara_mejora] = aptitud_vectores_de_prueba[mascara_mejora]

        aptitud_mejor_gen_actual = np.min(aptitud_poblacion)
        if aptitud_mejor_gen_actual < mejor_aptitud_global:
            mejor_aptitud_global = aptitud_mejor_gen_actual
            mejor_solucion_global = limite_inferior + poblacion_actual[np.argmin(aptitud_poblacion)] * (limite_superior - limite_inferior)
        
        historial_aptitud.append(mejor_aptitud_global)
        if gen_actual >= paciencia_convergencia:
            if abs(historial_aptitud[gen_actual] - historial_aptitud[gen_actual-paciencia_convergencia]) < tolerancia_convergencia:
                generaciones_ejecutadas = gen_actual + 1
                break

    solucion_final_escalada = limite_inferior + poblacion_actual[np.argmin(aptitud_poblacion)] * (limite_superior - limite_inferior)
    aptitud_final = np.min(aptitud_poblacion)
    
    return solucion_final_escalada, aptitud_final, generaciones_ejecutadas

# Parámetros
num_dims = 2
limites = [(-5, 5)] * num_dims # La función Ackley se evalúa a menudo en [-5, 5] o [-32, 32]
tam_pob = 50
param_F = 0.5  # Peso diferencial
param_CR = 0.7 # Probabilidad de cruce
max_gens = 500
semilla = 42 # Para reproducibilidad

print(f"Optimizando la Función de Ackley (N-Dimensiones = {num_dims})")
print(f"Límites: {limites[0]}")
print(f"Tamaño de Población: {tam_pob}, F: {param_F}, CR: {param_CR}, Máx Generaciones: {max_gens}\n")

# --- Random ---
tiempo_inicio_aleatorio = time.time()
solucion_aleatoria, aptitud_aleatoria, generaciones_aleatorias = ed_aleatorio_1_bin(
    funcion_ackley, limites, num_dims, tam_pob, param_F, param_CR, max_gens, semilla_aleatoria=semilla
)
tiempo_aleatorio = time.time() - tiempo_inicio_aleatorio

print("--- Resultados Random ---")
print(f"Tiempo tomado: {tiempo_aleatorio:.4f} segundos")
print(f"Generaciones hasta convergencia/máx: {generaciones_aleatorias}")
print(f"Mejor solución: {solucion_aleatoria}")
print(f"Valor de la función objetivo: {aptitud_aleatoria:.8f}\n")

# --- BEst ---
tiempo_inicio_mejor = time.time()
solucion_mejor, aptitud_mejor, generaciones_mejor = ed_mejor_1_bin(
    funcion_ackley, limites, num_dims, tam_pob, param_F, param_CR, max_gens, semilla_aleatoria=semilla
)
tiempo_mejor = time.time() - tiempo_inicio_mejor

print("--- Resultados Best ---")
print(f"Tiempo tomado: {tiempo_mejor:.4f} segundos")
print(f"Generaciones hasta convergencia/máx: {generaciones_mejor}")
print(f"Mejor solución: {solucion_mejor}")
print(f"Valor de la función objetivo: {aptitud_mejor:.8f}\n")

print("--- Comparación ---")
if aptitud_aleatoria < aptitud_mejor:
    print("Random encontró una solución mejor o igual.")
elif aptitud_mejor < aptitud_aleatoria:
    print("Best encontró una solución mejor.")
else:
    print("Ambos métodos encontraron soluciones con el mismo valor objetivo.")

if tiempo_aleatorio < tiempo_mejor:
    print("Random fue más rápido.")
elif tiempo_mejor < tiempo_aleatorio:
    print("Best fue más rápido.")
else:
    print("Ambos métodos tuvieron tiempos de ejecución similares.")

if generaciones_aleatorias < generaciones_mejor:
    print("Random convergió en menos generaciones.")
elif generaciones_mejor < generaciones_aleatorias:
    print("Best convergió en menos generaciones.")
else:
    print("Ambos métodos tomaron el mismo número de generaciones para converger (o alcanzar el máximo).")