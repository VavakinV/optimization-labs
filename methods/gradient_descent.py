import numpy as np

def func(x, y):
    # Функция сферы
    return x**2 + y**2

    # Функция Бута
    # return (x + 2*y - 7)**2 + (2*x + y - 5)**2

    # Функция Матьяса
    # return 0.26*(x**2 + y**2) - 0.48*x*y

    # Функция Изома
    # return -np.cos(x)*np.cos(y)*np.exp(-((x-np.pi)**2 + (y-np.pi)**2))

    # Функция Экли
    # return -20*np.exp(-0.2*np.sqrt(0.5*(x**2+y**2))) - np.exp(0.5*np.cos(2*np.pi*x) + np.cos(2*np.pi*y)) + np.e + 20

def gradient(x, y, h=1e-5):
    # Производная по x: (f(x+h, y) - f(x-h, y)  ) / (2h)
    df_dx = (func(x + h, y) - func(x - h, y)) / (2 * h)
    # Производная по y: (f(x, y+h) - f(x, y-h)) / (2h)
    df_dy = (func(x, y + h) - func(x, y - h)) / (2 * h)
    return np.array([df_dx, df_dy])

def optimize(x0, y0, learning_rate=0.1, epsilon=1e-6, epsilon1=1e-6, epsilon2=1e-6, max_iter=100):
    history = []
    current_point = np.array([x0, y0])
    
    for i in range(max_iter):   
        grad = gradient(*current_point)
        current_value = func(*current_point)
        
        if np.any(np.abs(grad) > 1e10):
            return history, False, "Функция расходится (норма градиента слишком большая)"
        
        grad_norm = np.linalg.norm(grad)
        
        history.append({
            'iteration': i+1,
            'x': current_point[0],
            'y': current_point[1],
            'f_value': current_value,
            'grad_norm': grad_norm
        })
        
        if grad_norm < epsilon1:
            return history, True, "Сошёлся (норма градиента меньше заданной точности)"
        
        old_point = current_point 
        current_point = current_point - learning_rate * grad
        modified_learning_rate = learning_rate

        # Вариант проверки 1
        while not(func(*current_point) - func(*old_point)) < 0:
            modified_learning_rate = modified_learning_rate / 2
            current_point = old_point - modified_learning_rate * grad

        # Вариант проверки 2
        # while not(abs(func(*current_point) - func(*old_point)) < epsilon*(grad_norm**2)):
        #     modified_learning_rate = modified_learning_rate / 2
        #     current_point = old_point - modified_learning_rate * grad

        if (np.linalg.norm(current_point-old_point) < epsilon2) and (np.linalg.norm(func(*current_point)-func(*old_point)) < epsilon2):
            return history, True, "Сошёлся (разница значений функции меньше заданной точности)" 
    
    return history, False, "Не сошёлся (достигнуто максимальное количество итераций)"