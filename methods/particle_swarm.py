import numpy as np

class Particle:
    def __init__(self, swarm):
        self._position = self.getInitPosition(swarm)

        self._localBestPosition = self._position[:]

        self._localBestValue = swarm.getFuncValue(self._position)

        self._velocity = self.getInitVelocity(swarm)


    @property
    def position(self):
        return self._position
    
    @property
    def velocity(self):
        return self._velocity
    

    def getInitPosition(self, swarm):
        return np.random.rand(swarm.dimension) * (swarm.maxvalues - swarm.minvalues) + swarm.minvalues
    

    def getInitVelocity(self, swarm):
        minval = -(swarm.maxvalues - swarm.minvalues)
        maxval = swarm.maxvalues - swarm.minvalues

        return np.random.rand(swarm.dimension) * (maxval - minval) + minval
    

    def nextIteration(self, swarm):
        random_currentPosition = np.random.rand(swarm.dimension)
        random_globalPosition = np.random.rand(swarm.dimension)

        velocityRatio = swarm.localVelocityRatio + swarm.globalVelocityRatio

        commonRatio = (2.0*swarm.currentVelocityRatio) / (np.abs(2.0 - velocityRatio - np.sqrt(velocityRatio**2 - 4.0*velocityRatio)))

        newVelocity1 = commonRatio * self._velocity 
        newVelocity2 = commonRatio * swarm.localVelocityRatio * random_currentPosition * (self._localBestPosition - self._position)
        newVelocity3 = commonRatio * swarm.globalVelocityRatio * random_globalPosition * (swarm.globalBestPosition - self._position)

        newVelocity = newVelocity1 + newVelocity2 + newVelocity3

        self._velocity = newVelocity

        self._position += self._velocity

        funcValue = swarm.getFuncValue(self._position)

        if funcValue < self._localBestValue:
            self._localBestPosition = self._position[:]
            self._localBestValue = funcValue

class Swarm:
    def __init__(self, func, swarmsize, minvalues, maxvalues, currentVelocityRatio, localVelocityRatio, globalVelocityRatio, penaltyRatio):
        self._func = func
        self._swarmsize = swarmsize
        self._minvalues = np.array(minvalues[:])
        self._maxvalues = np.array(maxvalues[:])
        self._currentVelocityRatio = currentVelocityRatio
        self._localVelocityRatio = localVelocityRatio
        self._globalVelocityRatio = globalVelocityRatio
        self._globalBestValue = None
        self._globalBestPosition = None
        self._penaltyRatio = penaltyRatio

        self._swarm = self.createSwarm()

    @property
    def func(self):
        return self._func

    @property
    def minvalues(self):
        return self._minvalues

    @property
    def maxvalues(self):
        return self._maxvalues

    @property
    def currentVelocityRatio(self):
        return self._currentVelocityRatio

    @property
    def localVelocityRatio(self):
        return self._localVelocityRatio

    @property
    def globalVelocityRatio(self):
        return self._globalVelocityRatio

    @property
    def globalBestPosition(self):
        return self._globalBestPosition

    @property
    def globalBestValue(self):
        return self._globalBestValue
    
    @property
    def penaltyRatio(self):
        return self._penaltyRatio
    
    @property
    def dimension(self):
        return len(self._minvalues)

    def getParticle(self, index):
        return self._swarm[index]

    def createSwarm(self):
        return [Particle(self) for _ in range(self._swarmsize)]
    
    def nextIteration(self):
        for particle in self._swarm:
            particle.nextIteration(self)
    
    def getFuncValue(self, position):
        result = self._func(*position)

        if (self._globalBestValue is None) or (result < self._globalBestValue):
            self._globalBestValue = result
            self._globalBestPosition = position[:]

        return result + self.getPenalty(position)
    
    def getPenalty(self, position):
        penalty1 = sum([self._penaltyRatio*abs(coord-minval) for coord, minval in zip(position, self.minvalues) if coord < minval])
        penalty2 = sum([self._penaltyRatio*abs(coord-maxval) for coord, maxval in zip(position, self.maxvalues) if coord > maxval])

        return penalty1 + penalty2
    

def optimize(func, maxIter, swarmsize, bounds, currentVelocityRatio, localVelocityRatio, globalVelocityRatio, penaltyRatio):

    # Инициализация параметров
    history = []

    swarm = Swarm(func, swarmsize, bounds[:][0], bounds[:][1], currentVelocityRatio, localVelocityRatio, globalVelocityRatio, penaltyRatio)

    for i in range(maxIter):
        swarm.nextIteration()
        history.append({
            'iteration': i+1,
            'x': swarm.globalBestPosition[0],
            'y': swarm.globalBestPosition[1],
            'f_value': swarm.globalBestValue
        })

    # Формирование результата
    converged = True
    message = "Оптимум найден" if converged else "Достигнуто максимальное количество итераций"
    
    return history, converged, message