import numpy as np
from sklearn.utils import check_random_state

def pso(fun, n_dim, lower_bound = 0, upper_bound = 1, swarm_size = 100, inertia_weight = 0.5, c1 = 2, c2 = 2, max_iters = 100, random_state = None):
	"""
	Optimizacion por enjambre de particulas
	fun: funcion objetivo a optimizar
	n_dim: int, numero de variables del problema de optimizacion 
	lower_bound: int, array, limite inferior de las variables 
	upper_bound: int, array, limite superior de las variables 
	swarm_size: int,numero de individuos en el enjambre 
	inertia_weight: float, peso de inercia
	c1, c2: float, componente cognitivo y social
	max_iters: int, numero maximo de iteraciones/generaciones }
	random_state: int, semilla de los numeros aleatorios
	"""
	# Inicializar manejador de numeros aleatorios
	random_state = check_random_state(random_state)
	# Crear el enjambre inicial e inicializar velocidades en cero
	swarm_position = random_state.uniform(size=(swarm_size,n_dim))
	swarm_velocity = np.zeros((swarm_size,n_dim))
	fit = [fun(np.subtract(upper_bound, lower_bound) * particle + lower_bound) for particle in swarm_position]
	fit = np.array(fit)
	# Definir cada particula como su mejor posicion
	personal_best =  {'position': swarm_position , 'quality': fit}
	#definir el vecindario aleatoriamente
	neighborhood = np.argsort(random_state.uniform(size=(swarm_size ,swarm_size)))[: ,:5] 
	# Guardar la mejor particula hasta el momento
	particle_best={'position': swarm_position[np.argmin(fit)], 'quality': np.min(fit )}
	for _ in range(max_iters):
		# Determinar la mejor particula del vecindario y actualizar la velocidad
		idx_best = np.argmin(fit[neighborhood], axis = 1)
		swarm_velocity *= inertia_weight
		swarm_velocity += c1 * random_state.uniform() * np.subtract(personal_best['position'], swarm_position)
		swarm_velocity += c2 * random_state.uniform() * np.subtract(swarm_position[idx_best], swarm_position)
		# Desplazar la particula
		swarm_position = np.clip(swarm_position + swarm_velocity, 0, 1)
		# Evaluando el fitness de la poblacion descendiente
		fit = [fun(np.subtract(upper_bound, lower_bound) * particle + lower_bound) for particle in swarm_position]
		fit = np.array(fit)
		# Comparar si mejora su mejor posicion
		is_improved = np.where(fit < personal_best['quality'], True, False)
		if np.any(is_improved):
			personal_best['position'][is_improved] = swarm_position[is_improved]
			personal_best['quality'][is_improved] = fit[is_improved]
		# Comparar si es la mejor global
		if np.min(fit) < particle_best['quality']:
			particle_best = {'position': swarm_position[np.argmin(fit)], 'quality': np.min(fit)}
		particle_best['position'] *= np.subtract(upper_bound , lower_bound)
		particle_best['position'] += lower_bound
		return np.add(np.subtract(upper_bound , lower_bound) * swarm_position , lower_bound), fit , particle_best



def func(x : np.ndarray, A : float):
    """Función de arrastre"""
    return A*len(x) + np.sum(x**2 - A*np.cos(2*np.pi*x))

def main():
	n_dim = 2
	lower_bound = -5.12
	upper_bound = 5.12
	swarm_size = 100
	A = 10	
 
	# Considerando solo componente cognitivo
	c1 = 2
	c2 = 0
	max_iters = 100
	random_state = 42
 
	# Definir la funcion objetivo
	fun = lambda x: func(x, A)
	# Ejecutar el algoritmo PSO
	swarm_position, fit, particle_best = pso(fun, n_dim, lower_bound, upper_bound, swarm_size, c1 = c1, c2 = c2, max_iters = max_iters, random_state = random_state)	
	print("Mejor posicion: ", particle_best['position'])
	print("Mejor fitness: ", particle_best['quality'])

if __name__ == "__main__":
	main()



