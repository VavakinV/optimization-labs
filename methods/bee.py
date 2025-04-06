import random, math

class Bee:
    def __init__(self, func, radius):
        self.minval = [-x for x in radius]
        self.maxval = [x for x in radius]

        self.position = [random.uniform(self.minval[i], self.maxval[i]) for i in range(2)]

        self.fitness = 0.0

        self.func = func
    
    def calcFitness(self):
        self.fitness = self.func(*self.position)

    def otherPatch(self, bee_list, range_list):
        if len(bee_list) == 0:
            return True
        
        self.calcFitness()
        for bee in bee_list:
            bee.calcFitness()
            pos = bee.getPosition()
            for n in range(len(self.position)):
                if abs(self.position[n] - pos[n]) > range_list[n]:
                    return True
                
        return False
    
    def getPosition(self):
        return [val for val in self.position]
    
    def goto(self, otherpos, range_list):
        self.position = [otherpos[n] + random.uniform(-range_list[n], range_list[n]) for n in range(len(otherpos))]

        self.checkPosition()

        self.calcFitness()

    def gotorandom(self):
        self.position = [random.uniform(self.minval[n], self.maxval[n]) for n in range(len(self.position))]

        self.checkPosition()

        self.calcFitness()

    def checkPosition(self):
        for n in range(len(self.position)):
            if self.position[n] < self.minval[n]:
                self.position[n] = self.minval[n]
            
            elif self.position[n] > self.maxval[n]:
                self.position[n] = self.maxval[n]

class Hive:
    def __init__(self, scoutbee_count, selectedbee_count, bestbee_count, selectedsites_count, bestsites_count, radius, func):
        self.scoutbee_count = scoutbee_count
        self.selectedbee_count = selectedbee_count
        self.bestbee_count = bestbee_count
        self.selectedsites_count = selectedsites_count
        self.bestsites_count = bestsites_count
        self.radius = radius
        self.func = func

        self.best_position = None

        self.best_fitness = math.inf

        bee_count = scoutbee_count + selectedbee_count * selectedsites_count + bestbee_count * bestsites_count
        self.swarm = [Bee(func, radius) for _ in range(bee_count)]

        self.bestsites = []
        self.selectedsites = []

        self.swarm.sort(key=lambda x: x.fitness)
        self.best_position = self.swarm[0].getPosition()
        self.best_fitness = self.swarm[0].fitness

    def sendBees(self, position, index, count):
        for i in range(count):
            if index == len(self.swarm):
                break
            
            bee = self.swarm[index]
            if not(bee in self.bestsites) and not(bee in self.selectedsites):
                bee.goto(position, self.radius)

            index += 1

        return index 
    
    def nextIteration(self):
        # Пересчитываем fitness для всех пчел перед сортировкой
        for bee in self.swarm:
            bee.calcFitness()
        
        self.swarm.sort(key=lambda x: x.fitness)
        self.best_position = self.swarm[0].getPosition()
        self.best_fitness = self.swarm[0].fitness

        self.bestsites = [self.swarm[0]]
        self.selectedsites = []

        # Выбираем best sites
        curr_index = 1
        while curr_index < len(self.swarm) and len(self.bestsites) < self.bestsites_count:
            bee = self.swarm[curr_index]
            if bee.otherPatch(self.bestsites, self.radius):
                self.bestsites.append(bee)
            curr_index += 1

        # Выбираем selected sites
        while curr_index < len(self.swarm) and len(self.selectedsites) < self.selectedsites_count:
            bee = self.swarm[curr_index]
            if bee.otherPatch(self.bestsites, self.radius) and bee.otherPatch(self.selectedsites, self.radius):
                self.selectedsites.append(bee)
            curr_index += 1

        # Отправляем пчел
        bee_index = len(self.bestsites) + len(self.selectedsites)
        
        # Лучшие участки
        for best_bee in self.bestsites:
            for _ in range(self.bestbee_count):
                if bee_index >= len(self.swarm):
                    break
                self.swarm[bee_index].goto(best_bee.getPosition(), self.radius)
                bee_index += 1

        # Выбранные участки
        for sel_bee in self.selectedsites:
            for _ in range(self.selectedbee_count):
                if bee_index >= len(self.swarm):
                    break
                self.swarm[bee_index].goto(sel_bee.getPosition(), self.radius)
                bee_index += 1

        # Остальные - разведчики
        while bee_index < len(self.swarm):
            self.swarm[bee_index].gotorandom()
            bee_index += 1     

def optimize(func, maxiter, scoutbee_count, selectedbee_count, bestbee_count, bestsites_count, selectedsites_count, radius, koeff, tolerance, globaltolerance):
    history = []
    hive = Hive(scoutbee_count, selectedbee_count, bestbee_count, selectedsites_count, bestsites_count, radius, func)

    best_value = math.inf
    tolerance_counter = 0
    converged = False
    message = "Достигнуто максимальное количество итераций"

    for i in range(maxiter):
        hive.nextIteration()

        if abs(hive.best_fitness - best_value) > 1e-5:
            best_value = hive.best_fitness
            hive.radius = [hive.radius[i] * koeff for i in range(len(hive.radius))]
            tolerance_counter = 0
        else:
            tolerance_counter += 1
            if tolerance_counter >= tolerance:
                hive.radius = [hive.radius[i] / koeff for i in range(len(hive.radius))]
                tolerance_counter = 0
                globaltolerance -= 1
                if globaltolerance == 0:
                    converged = True
                    message = "Достигнут предел расширений"
                    break

        history.append({
            'iteration': i+1,
            'x': hive.best_position[0],
            'y': hive.best_position[1],
            'f_value': hive.best_fitness
        })

    converged = True
    message = "Оптимум найден" if converged else "Достигнуто максимальное количество итераций"
    
    return history, converged, message