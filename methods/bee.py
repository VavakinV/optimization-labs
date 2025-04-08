import random, math

class Bee:
    def __init__(self, func, x_min, x_max, y_min, y_max):
        self.minval = [x_min, y_min]
        self.maxval = [x_max, y_max]

        self.position = [random.uniform(self.minval[i], self.maxval[i]) for i in range(2)]

        self.fitness = 0.0

        self.func = func
    
    def calcFitness(self):
        self.fitness = self.func(*self.position)

    def otherPatch(self, bee_list, radius):
        if len(bee_list) == 0:
            return True
        
        self.calcFitness()
        for bee in bee_list:
            bee.calcFitness()
            pos = bee.getPosition()
            for i in range(2):
                if abs(self.position[i] - pos[i]) > radius:
                    return True
                
        return False
    
    def getPosition(self):
        return [val for val in self.position]
    
    def goto(self, otherpos, radius):
        self.position = [otherpos[n] + random.uniform(-radius, radius) for n in range(len(otherpos))]

        self.checkPosition()

        self.calcFitness()

    def gotorandom(self):
        self.position = [random.uniform(self.minval[i], self.maxval[i]) for i in range(2)]

        self.checkPosition()

        self.calcFitness()

    def checkPosition(self):
        for n in range(len(self.position)):
            if self.position[n] < self.minval[n]:
                self.position[n] = self.minval[n]
            
            elif self.position[n] > self.maxval[n]:
                self.position[n] = self.maxval[n]

class Hive:
    def __init__(self, scoutbee_count, selectedbee_count, bestbee_count, selectedsites_count, bestsites_count, radius, func, x_min, x_max, y_min, y_max):
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
        self.swarm = [Bee(func, x_min, x_max, y_min, y_max) for _ in range(bee_count)]

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
        for bee in self.swarm:
            bee.calcFitness()
        
        self.swarm.sort(key=lambda x: x.fitness)
        self.best_position = self.swarm[0].getPosition()
        self.best_fitness = self.swarm[0].fitness

        self.bestsites = [self.swarm[0]]
        self.selectedsites = []

        curr_index = 1
        while curr_index < len(self.swarm) and len(self.bestsites) < self.bestsites_count:
            bee = self.swarm[curr_index]
            if bee.otherPatch(self.bestsites, self.radius):
                self.bestsites.append(bee)
            curr_index += 1

        while curr_index < len(self.swarm) and len(self.selectedsites) < self.selectedsites_count:
            bee = self.swarm[curr_index]
            if bee.otherPatch(self.bestsites, self.radius) and bee.otherPatch(self.selectedsites, self.radius):
                self.selectedsites.append(bee)
            curr_index += 1

        bee_index = 1
        
        for best_bee in self.bestsites:
            bee_index = self.sendBees(best_bee.getPosition(), bee_index, self.bestbee_count)

        for sel_bee in self.selectedsites:
            bee_index = self.sendBees(sel_bee.getPosition(), bee_index, self.selectedbee_count)

        for bee in self.swarm[bee_index:-1]:
            bee.gotorandom()  

def optimize(func, maxiter, scoutbee_count, selectedbee_count, bestbee_count, bestsites_count, selectedsites_count, radius, koeff, tolerance, globaltolerance, x_min, x_max, y_min, y_max):
    history = []
    hive = Hive(scoutbee_count, selectedbee_count, bestbee_count, selectedsites_count, bestsites_count, radius, func, x_min, x_max, y_min, y_max)

    best_value = math.inf
    tolerance_counter = 0
    converged = False
    message = "Достигнуто максимальное количество итераций"

    for i in range(maxiter):
        hive.nextIteration()

        if abs(hive.best_fitness - best_value) > 1e-5:
            best_value = hive.best_fitness
            hive.radius = hive.radius * koeff
            tolerance_counter = 0
        else:
            tolerance_counter += 1
            if tolerance_counter >= tolerance:
                hive.radius = hive.radius / koeff
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