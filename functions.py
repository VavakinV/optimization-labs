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
        case "rastrigin":
            return lambda x, y: 20 + (x**2 - 10*np.cos(2*np.pi*x)) + (y**2 - 10*np.cos(2*np.pi*y))
        case "goldstein_price":
            return lambda x, y: (1 + ((x+y+1)**2)*(19-14*x+3*(x**2)-14*y+6*x*y+3*(y**2)))*(30 + ((2*x-3*y)**2)*(18-32*x+12*(x**2)+48*y-36*x*y+27*(y**2)))
        case "cross_in_tray":
            return lambda x, y: -0.0001 * ((abs(np.sin(x)*np.sin(y)*np.exp(abs(100 - (np.sqrt(x**2+y**2)/np.pi)))) + 1)**0.1)