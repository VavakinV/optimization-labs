import numpy as np

def functions(function_name):
    s = function_name.lower()
    match s:
        case "rosenbrock":
            return lambda x, y: (1-x)**2 + 100*((y-x**2)**2)
        case "bukin":
            return lambda x, y: 100*np.sqrt(abs(y-0.01*(x**2))) + 0.01*abs(x+10)
        case "himmelblau":
            return lambda x, y: (x**2 + y -11)**2 + (x + y**2 - 7)**2
        case "isom":
            return lambda x, y: -np.cos(x)*np.cos(y)*np.exp(-((x-np.pi)**2 + (y-np.pi)**2))

def optimize(objective_func, bounds, used_methods={"crossover": True, "mutation": True}, population_size=50, 
             crossover_prob=0.8, mutation_prob=0.1, mutation_parameter=3,
             max_iter=100, tol=1e-6, patience=25):

    # Инициализация параметров
    history = []

    # 1. Генерация начальной популяции
    population = np.zeros((population_size, 2))
    population[:, 0] = np.random.uniform(bounds[0][0], bounds[0][1], population_size)
    population[:, 1] = np.random.uniform(bounds[1][0], bounds[1][1], population_size)
    
    best_fitness = np.inf
    no_improve = 0

    recombination_parameter = 0.25
    
    for iteration in range(max_iter):
        # 2. Вычисление пригодности
        objective_values = np.array([objective_func(ind[0], ind[1]) for ind in population])
        
        # Сохранение лучшей особи
        best_idx = np.argmin(objective_values)
        current_best = population[best_idx]
        current_value = objective_values[best_idx]
        
        # Критерий остановки
        if abs(current_value - best_fitness) < tol:
            no_improve += 1
            if no_improve >= patience:
                break
        else:
            if current_value < best_fitness:
                no_improve = 0
                best_fitness = current_value
        
        # Запись в историю
        history.append({
            'iteration': iteration+1,
            'x': current_best[0],
            'y': current_best[1],
            'f_value': current_value
        })
        
        fitness = 1 / (1+objective_values)
        fitness_sum = fitness.sum()
        if fitness_sum == 0:
            probabilities = np.ones(population_size) / population_size
        else:
            probabilities = fitness / fitness_sum
        
        for i in range(len(probabilities)):
            if np.isnan(probabilities[i]):
                probabilities[i] = 0

        # 3-6. Генерация нового поколения
        temporary_population = []
        while len(temporary_population) < population_size:
            # Отбор (метод рулетки)
            parents = population[np.random.choice(
                population_size, 2, p=probabilities, replace=False)]
            
            # Рекомбинация (линейная)
            if used_methods['crossover']:
                rec_coeffs = np.random.uniform(-recombination_parameter, 1+recombination_parameter, 2)
                if np.random.rand() < crossover_prob:
                    child1 = parents[0] + rec_coeffs[0]*(parents[1]-parents[0])
                    child2 = parents[0] + rec_coeffs[1]*(parents[1]-parents[0])
                else:
                    child1, child2 = parents[0], parents[1]
            else:
                child1, child2 = parents[0], parents[1]
            
            for child in [child1, child2]:
                # Мутация (мутация для вещественных особей)
                if used_methods['mutation']:
                    if np.random.rand() < mutation_prob:
                        delta = 0
                        for i in range(1, mutation_parameter+1):
                            delta += 2**(-i) * (1 if np.random.rand()<(1/mutation_parameter) else 0)
                        child[0] += delta+0.5*(bounds[0][1]-bounds[0][0]) * ((-1) if np.random.rand()<=0.5 else 1)
                        child[1] += delta+0.5*(bounds[1][1]-bounds[1][0]) * ((-1) if np.random.rand()<=0.5 else 1)
                    
                    # Проверка границ
                    child[0] = np.clip(child[0], bounds[0][0], bounds[0][1])
                    child[1] = np.clip(child[1], bounds[1][0], bounds[1][1])
                temporary_population.append(child)

        population = np.array(temporary_population)

    # Формирование результата
    converged = True
    message = "Оптимум найден" if converged else "Достигнуто максимальное количество итераций"
    
    return history, converged, message