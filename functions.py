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