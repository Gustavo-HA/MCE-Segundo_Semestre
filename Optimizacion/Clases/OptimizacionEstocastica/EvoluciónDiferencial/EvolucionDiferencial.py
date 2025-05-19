import numpy as np
from sklearn.utils import check_random_state
def differential_evolution(fun, n_dim, lower_bound = 0, upper_bound = 1, population_size = 100, CR = 0.5, F = 0.5,max_iters = 100, random_state = None):
	"""
	Evolucion diferencial
	fun: funcion objetivo a optimizar
	n_dim: int, numero de variables del problema de optimizacion lower_bound: int, array, limite inferior de las variables upper_bound: int, array, limite superior de las variables population_size: int, numero de individuos en la poblacion CR: float , tasa de cruza
	F: float , peso diferencial
	max_iters: int, numero maximo de iteraciones/generaciones random_state: int, semilla de los numeros aleatorios
	"""
	# Inicializar manejador de numeros aleatorios
	random_state = check_random_state(random_state)
	# Crear la poblacion inicial
	population = random_state.uniform(size = (population_size ,n_dim))
	# Evaluando la aptutud de cada individuo en la poblacion
	fit = [fun(np.subtract(upper_bound, lower_bound) * ind + lower_bound) for ind in population]
	for _ in range(max_iters):
		r1, r2, r3 = np.argsort(random_state.uniform(size = (population_size, population_size))).T[:3]
		# Creando el vector de prueba usando la mutacion direccional
		trial = np.clip(population[r1] + F * (population[r2] - population[r3]),0 ,1)
		## Cruza binomial
		# Definiendo los puntos de cruza
		cross_points = random_state.uniform(size = (population_size, n_dim)) < CR
		is_any_true = np.all(~cross_points , axis = 1)
		if np.any(is_any_true):
			cross_points[is_any_true,np.random.randint(n_dim, size=(is_any_true.sum()))]= True
			offspring = population.copy()
			offspring[cross_points] = trial[cross_points]
			# Evaluando el fitness de la poblacion descendiente
			fit_offspring = [fun(np.subtract(upper_bound, lower_bound) * ind + lower_bound) for ind in offspring]
			# Seleccionando los individuos de la siguiente generacion
			is_better = np.where(fit_offspring < fit, True, False)
			if np.any(is_better):
				population[is_better] = offspring[is_better]
				fit = np.where(is_better , fit_offspring , fit)
			return np.add(np.subtract(upper_bound , lower_bound) * population , lower_bound), fit