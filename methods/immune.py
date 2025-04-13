import numpy as np

def optimize(func, max_iter, population_size, x_min, x_max, y_min, y_max, nb, nc, nd, mutation, tolerance_steps):
    
    dim = 2
    population = np.random.uniform(low=[x_min, y_min], 
                                 high=[x_max, y_max],
                                 size=(population_size, dim))
    
    history = []
    best_fitness = np.inf
    best_position = None

    delta_mutation = mutation/max_iter
    mutation_rate = mutation

    tolerance = 1e-16
    no_improvement_steps = 0
    
    for iteration in range(max_iter):
        fitness = np.array([func(*ind) for ind in population])
        
        best_indices = np.argsort(fitness)[:nb]
        Sb = population[best_indices].copy()

        Sm = np.repeat(Sb, nc, axis=0)
        
        mutation = mutation_rate * np.random.uniform(-0.5, 0.5, Sm.shape)
        Sm = Sm + mutation

        Sm[:, 0] = np.clip(Sm[:, 0], x_min, x_max)
        Sm[:, 1] = np.clip(Sm[:, 1], y_min, y_max)

        Sm_fitness = np.array([func(*ind) for ind in Sm])
        best_Sm_indices = np.argsort(Sm_fitness)[:nd]
        Sm = Sm[best_Sm_indices]
        
        combined_population = np.vstack((population, Sm))
        combined_fitness = np.array([func(*ind) for ind in combined_population])
        
        best_combined_indices = np.argsort(combined_fitness)[:population_size]
        population = combined_population[best_combined_indices]
        
        current_best_fitness = combined_fitness[best_combined_indices[0]]
        current_best_position = population[0]

        mutation_rate -= delta_mutation

        if abs(best_fitness - current_best_fitness) < tolerance:
            no_improvement_steps += 1
        else:
            no_improvement_steps = 0
        if no_improvement_steps >= tolerance_steps:
            history.append({
                'iteration': iteration+1,
                'x': best_position[0],
                'y': best_position[1],
                'f_value': best_fitness
            })
            converged = True
            message = "Оптимум найден"
            
            return history, converged, message

        if current_best_fitness < best_fitness:
            best_fitness = current_best_fitness
            best_position = current_best_position
        
        history.append({
            'iteration': iteration+1,
            'x': best_position[0],
            'y': best_position[1],
            'f_value': best_fitness
        })

    converged = False
    message = "Достигнуто максимальное количество итераций"
    
    return history, converged, message