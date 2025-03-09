from scipy.optimize import minimize, linprog
import numpy as np


# целевая функция 2-ух переменных
def objective(x, coeffs):
    x1, x2 = x[0], x[1]
    return coeffs[0] * x1 ** 2 + coeffs[1] * x2 ** 2 + coeffs[2] * x1 * x2 + coeffs[3] * x1 + coeffs[4] * x2

def objective_param(coeffs):
    return lambda x1, x2: coeffs[0] * x1 ** 2 + coeffs[1] * x2 ** 2 + coeffs[2] * x1 * x2 + coeffs[3] * x1 + coeffs[4] * x2

# ограничения (сколько хочешь + x1>=0, x2>=0)
def constraints(coeffs_con):
    cons = []
    for i in range(0, len(coeffs_con), 3):
        cons.append({'type': 'ineq',
                     'fun': lambda x: coeffs_con[i + 2] - (coeffs_con[i] * x[0] + coeffs_con[i + 1] * x[1])})

    # Ограничения для x1 >= 0 и x2 >= 0
    cons.append({'type': 'ineq', 'fun': lambda x: x[0]})  # x1 >= 0
    cons.append({'type': 'ineq', 'fun': lambda x: x[1]})  # x2 >= 0

    return cons

def optimize(x0, coeffs_obj, coeffs_con, type):
    if type == "minimize":
        result = minimize(objective, x0, args=(coeffs_obj), constraints=constraints(coeffs_con))
        f_value = result.fun
    elif type == "maximize":
        result = minimize(lambda x, coeffs: -objective(x, coeffs), x0, args=(coeffs_obj), constraints=constraints(coeffs_con))
        f_value = -result.fun

    return [{
            'iteration': 1,
            'x': result.x[0],
            'y': result.x[1],
            'f_value': f_value,
            'grad_norm': np.nan
    }
    ], True, "Минимум функции найден. Сообщение: "+result.message if type=="minimize" else "Максимум функции найден. Сообщение: "+result.message