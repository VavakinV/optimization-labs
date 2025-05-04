import random, copy
import numpy as np

class Bacteria:
    def __init__(self, func, x_min, x_max, y_min, y_max):
        self.minval = [x_min, y_min]
        self.maxval = [x_max, y_max]

        self.position = np.array([random.uniform(self.minval[i], self.maxval[i]) for i in range(2)])

        self.func = func

        self.health = func(*self.position)
        self.func_value = self.health

        self.improved_last_step = True

        self.movement_vector = np.random.rand(2)
        self.movement_vector_norm = np.linalg.norm(self.movement_vector)

    def move(self, chemotaxis_step):
        if not(self.improved_last_step):
            self.movement_vector = np.random.rand(2)
            self.movement_vector_norm = np.linalg.norm(self.movement_vector)

        self.position += chemotaxis_step * self.movement_vector / self.movement_vector_norm

        self.position[0] = np.clip(self.position[0], self.minval[0], self.maxval[0])
        self.position[1] = np.clip(self.position[1], self.minval[1], self.maxval[1])

        new_func_value = self.func(*self.position)
        self.health += new_func_value

        if new_func_value > self.func_value:
            self.improved_last_step = False
        else:
            self.improved_last_step = True
        
        self.func_value = new_func_value


class BacterialPopulation:
    def __init__(self, func, x_min, x_max, y_min, y_max, population_count, n_chemotaxis, n_reproduction, n_elimination, chemotaxis_step, elimination_threshold, elimination_probabilty, elimination_count):
        self.population = [Bacteria(func, x_min, x_max, y_min, y_max) for _ in range(population_count)]

        self.chemotaxis_step = chemotaxis_step

        self.n_chemotaxis = n_chemotaxis
        self.n_reproduction = n_reproduction
        self.n_elimination = n_elimination

        self.chemotaxis_step_reduction = chemotaxis_step / n_chemotaxis

        self.elimination_threshold = elimination_threshold

        self.elimination_probabilty = elimination_probabilty
        self.elimination_count = elimination_count

        self.best_func = self.population[0].func_value
        self.best_pos = self.population[0].position.copy()

        self.chemotaxiss_completed = 0
        self.reproductions_completed = 0
        self.eliminations_completed = 0

    
    def chemotaxis(self):
        if self.chemotaxiss_completed >= self.n_chemotaxis:
            return

        for bacteria in self.population:
            bacteria.move(self.chemotaxis_step)

        self.chemotaxis_step -= self.chemotaxis_step_reduction
        
        self.chemotaxiss_completed += 1

    def reproduction(self):
        if self.reproductions_completed >= self.n_reproduction:
            return

        self.population.sort(key=lambda x: x.health)

        new_population = []

        for i in range(len(self.population)//2):
            new_population.extend([copy.deepcopy(self.population[i]), copy.deepcopy(self.population[i])])
        
        self.population = new_population

        self.reproductions_completed += 1

    def elimination(self):
        q = random.random()
        if (self.reproductions_completed < self.elimination_threshold) or (q <= self.elimination_probabilty) or (self.eliminations_completed >= self.n_elimination):
            return
        
        for _ in range(self.elimination_count):
            i = np.random.randint(0, len(self.population))
            self.population[i].position = np.array([random.uniform(self.population[i].minval[j], self.population[i].maxval[j]) for j in range(2)])

        self.eliminations_completed += 1

    def next_step(self):
        self.chemotaxis()
        self.reproduction()
        self.elimination()

        for bacteria in self.population:
            if bacteria.func_value < self.best_func:
                self.best_func = bacteria.func_value
                self.best_pos = bacteria.position.copy()


def optimize(func, x_min, x_max, y_min, y_max, population_count, n_chemotaxis, n_reproduction, n_elimination, chemotaxis_step, elimination_threshold, elimination_probabilty, elimination_count):
    history = []
    population = BacterialPopulation(func, x_min, x_max, y_min, y_max, population_count, n_chemotaxis, n_reproduction, n_elimination, chemotaxis_step, elimination_threshold, elimination_probabilty, elimination_count)

    converged = False
    message = "Достигнуто максимальное количество итераций"

    for i in range(n_chemotaxis):
        population.next_step()

        history.append({
            'iteration': i+1,
            'x': population.best_pos[0],
            'y': population.best_pos[1],
            'f_value': population.best_func
        })

    converged = True
    message = "Оптимум найден"
    
    return history, converged, message


