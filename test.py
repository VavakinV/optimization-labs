import dash
from dash import dcc, html, Input, Output, State, dash_table
import plotly.graph_objs as go
import numpy as np
import dash_bootstrap_components as dbc

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

def gradient_descent(x0, y0, learning_rate=0.1, epsilon=1e-6, epsilon1=1e-6, epsilon2=1e-6, max_iter=100):
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

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

app.layout = dbc.Container([
    html.H1("Градиентный спуск с постоянным шагом", className='mt-3'),
    
    dbc.Alert(id='final-result', color="success", className='mt-3'),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Параметры", className="card-title"),
                    dbc.InputGroup([
                        dbc.InputGroupText("X₀"),
                        dbc.Input(id='x0-input', type='number', value=0)
                    ], className='mb-2'),
                    
                    dbc.InputGroup([
                        dbc.InputGroupText("Y₀"),
                        dbc.Input(id='y0-input', type='number', value=0)
                    ], className='mb-2'),
                    
                    dbc.InputGroup([
                        dbc.InputGroupText("Шаг"),
                        dbc.Input(id='lr-input', type='number', value=0.1, step=0.01)
                    ], className='mb-2'),

                    dbc.InputGroup([
                        dbc.InputGroupText("Точность ε (проверка убывания)"),
                        dbc.Input(id='epsilon-input', type='number', value=1e-4, step=1e-6)
                    ], className='mb-2'),
                    
                    dbc.InputGroup([
                        dbc.InputGroupText("Точность ε1 (норма градиента в точке)"),
                        dbc.Input(id='epsilon1-input', type='number', value=1e-4, step=1e-6)
                    ], className='mb-2'),

                    dbc.InputGroup([
                        dbc.InputGroupText("Точность ε2 (разность значений функций)"),
                        dbc.Input(id='epsilon2-input', type='number', value=1e-4, step=1e-6)
                    ], className='mb-2'),
                    
                    dbc.InputGroup([
                        dbc.InputGroupText("Макс. итераций"),
                        dbc.Input(id='max-iter-input', type='number', value=100)
                    ], className='mb-2'),
                    
                    dbc.Button("Запустить", id='run-button', color='primary', className='mt-3')
                ])
            ])
        ], md=4),
        
        dbc.Col([
            dcc.Graph(id='3d-plot'),
            html.Div(id='results-table', className='mt-3')
        ], md=8)
    ])
], fluid=True)

@app.callback(
    [Output('3d-plot', 'figure'),
     Output('results-table', 'children'),
     Output('final-result', 'children'),
     Output('final-result', 'color')],
    [Input('run-button', 'n_clicks')],
    [State('x0-input', 'value'),
     State('y0-input', 'value'),
     State('lr-input', 'value'),
     State('epsilon-input', 'value'),
     State('epsilon1-input', 'value'),
     State('epsilon2-input', 'value'),
     State('max-iter-input', 'value')]
)
def update_plot_and_table(n_clicks, x0, y0, lr, epsilon, epsilon1, epsilon2, max_iter):
    if None in [x0, y0, lr, epsilon, epsilon1, epsilon2, max_iter]:
        return go.Figure(), "Пожалуйста, заполните все поля", "", "danger"
    
    history, converged, status_message = gradient_descent(x0, y0, lr, epsilon, epsilon1, epsilon2, max_iter)
    
    if history:
        final = history[-1]
        result_message = [
            html.Strong("Результаты оптимизации:"),
            html.Br(),
            f"Финальная точка: ({round(final['x'], 4)}, {round(final['y'], 4)})",
            html.Br(),
            f"Значение функции: {round(final['f_value'], 4)}",
            html.Br(),
            f"Итераций выполнено: {final['iteration']}",
            html.Br(),
            f"Состояние: {status_message}"
        ]
        color = "success" if converged else "warning"
    else:
        result_message = "Не удалось выполнить оптимизацию"
        color = "danger"
    
    x = np.linspace(-10, 10, 100)
    y = np.linspace(-10, 10, 100)
    X, Y = np.meshgrid(x, y)
    Z = func(X, Y)
    
    trajectory_x = [point['x'] for point in history]
    trajectory_y = [point['y'] for point in history]
    trajectory_z = [point['f_value'] for point in history]
    
    fig = go.Figure(data=[
        go.Surface(x=X, y=Y, z=Z, colorscale='Viridis', opacity=0.8),
        go.Scatter3d(
            x=trajectory_x,
            y=trajectory_y,
            z=trajectory_z,
            mode='lines+markers',
            marker=dict(size=5, color='red'),
            line=dict(color='red', width=2)
        )
    ])
    
    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='f(x, y)'
        ),
        margin=dict(l=0, r=0, b=0, t=30)
    )
    
    columns = [
        {'name': 'Итерация', 'id': 'iteration'},
        {'name': 'X', 'id': 'x'},
        {'name': 'Y', 'id': 'y'},
        {'name': 'f(x,y)', 'id': 'f_value'},
        {'name': 'Норма градиента', 'id': 'grad_norm'}
    ]
    
    formatted_history = [
        {
            'iteration': item['iteration'],
            'x': round(item['x'], 4),
            'y': round(item['y'], 4),
            'f_value': round(item['f_value'], 4),
            'grad_norm': round(item['grad_norm'], 4)
        }
        for item in history
    ]

    table = dash_table.DataTable(
        id='results-datatable',
        columns=columns,
        data=formatted_history,
        page_size=10,
        style_table={'overflowX': 'auto'},
        style_cell={'textAlign': 'center'},
        style_header={'fontWeight': 'bold'}
    )
    
    return fig, table, result_message, color

if __name__ == '__main__':
    app.run_server(debug=True)