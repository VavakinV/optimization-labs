import numpy as np

def functions(function_name):
    s = function_name.lower()
    match s:
        case "rosenbrock":
            return lambda x, y: (1-x)**2 + 100*((y-x**2)**2)

def optimize(objective_func, bounds, used_methods={"crossover": True, "mutation": True}, population_size=50, 
             crossover_prob=0.8, mutation_prob=0.1, mutation_parameter=20,
             max_iter=100, temperature=100, tol=1e-6, patience=25):

    # Инициализация параметров
    history = []

    # 1. Генерация начальной популяции
    population = np.zeros((population_size, 2))
    population[:, 0] = np.random.uniform(bounds[0][0], bounds[0][1], population_size)
    population[:, 1] = np.random.uniform(bounds[1][0], bounds[1][1], population_size)
    
    best_fitness = -np.inf
    no_improve = 0
    
    for iteration in range(max_iter):
        # 2. Вычисление пригодности
        fitness = np.array([objective_func(ind[0], ind[1]) for ind in population])
        
        # Сохранение лучшей особи
        best_idx = np.argmin(fitness)
        current_best = population[best_idx]
        current_value = fitness[best_idx]
        
        # Критерий остановки
        if abs(current_value - best_fitness) < tol:
            no_improve += 1
            if no_improve >= patience:
                break
        else:
            no_improve = 0
            best_fitness = current_value
        
        # Запись в историю
        history.append({
            'iteration': iteration+1,
            'x': current_best[0],
            'y': current_best[1],
            'f_value': current_value
        })
        
        # 3-6. Генерация нового поколения
        temporary_population = []
        while len(temporary_population) < population_size:
            # Отбор (метод рулетки)
            parents = population[np.random.choice(
                population_size, 2, p=fitness/fitness.sum(), replace=False)]
            
            # Скрещивание (метод BLX-α)
            if used_methods['crossover']:
                if np.random.rand() < crossover_prob:
                    alpha = np.random.rand()
                    child1 = alpha * parents[0] + (1-alpha) * parents[1]
                    child2 = alpha * parents[1] + (1-alpha) * parents[0]
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
        
        # Селекция (метод Больцмана)
        new_population = []
        while len(new_population) < population_size:
            i1, i2 = np.random.choice(len(temporary_population), 2)
            indiv1, indiv2 = temporary_population[i1], temporary_population[i2]
            p = 1/(1+np.exp((objective_func(*indiv1)-objective_func(*indiv2))/temperature))
            if p > np.random.rand():
                new_population.append(indiv1)
            else:
                new_population.append(indiv2)

        population = np.array(new_population)[:population_size]

    # Формирование результата
    converged = no_improve >= patience
    message = "Оптимум найден" if converged else "Не сошёлся (достигнуто максимальное количество итераций)"
    
    return history, converged, message