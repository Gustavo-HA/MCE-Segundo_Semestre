import numpy as np
from sklearn.utils import check_random_state

def random_search(fun, n_dim, lower_bound = 0, upper_bound = 1, max_iters = 1000, random_state = None):
	"""
	Busqued aleatoria pura
	fun: funcion objetivo a optimizar
	n_dim: int, numero de variables en la funcion objetivo
	lower_bound: int, array, limite inferior que puede tomar cada variable upper_bound: int, array, limite superior que puede tomar cada variable max_iters: int, maximo numero de iteraciones
	random_state: manejador de numeros aleatorios
	"""
	# Inicializar manejador de numeros aleatorios
	random_state = check_random_state(random_state)
	# Inicilizar valores
	best_val = np.infty
	best_sol = None
	# Busqueda aleatoria
	for _ in range(max_iters):
		candidate = random_state.uniform(low = lower_bound , high =(n_dim))
		obj_val = fun(candidate) 
		if obj_val < best_val:
			best_val = obj_val
			best_sol = candidate 
	return best_val , best_sol


fun = lambda w: w**2
x,y=random_search(fun,1)
print("best val:",x,"best sol:",y)