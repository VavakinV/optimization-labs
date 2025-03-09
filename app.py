import dash
from dash import dcc, html, Input, Output, State, dash_table
import plotly.graph_objs as go
import numpy as np
import dash_bootstrap_components as dbc

from methods import gradient_descent, simplex_method

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True, prevent_initial_callbacks='initial_duplicate')
server = app.server

methods = {
    "Градиентный спуск": gradient_descent,
    "Симплекс-метод": simplex_method,
}

app.layout = dbc.Container([
    html.H1("Методы оптимизации", className='mt-3'),
    
    dbc.Alert(id='final-result', color="success", className='mt-3'),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Выбор метода", className="card-title"),
                    dcc.Dropdown(
                        id="method-dropdown",
                        options=[{"label": name, "value": name} for name in methods.keys()],
                        value="Градиентный спуск",
                        clearable=False
                    ),
                    html.Div(id='method-params')
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
    Output('method-params', 'children'),
    Input('method-dropdown', 'value')
)
def update_params(method_name):
    if method_name == "Градиентный спуск":
        return html.Div([
            dbc.InputGroup([
                    dbc.InputGroupText("X₀"),
                    dbc.Input(id='gradient-x0-input', type='number', value=0)
                ], className='mb-2'),
                
                dbc.InputGroup([
                    dbc.InputGroupText("Y₀"),
                    dbc.Input(id='gradient-y0-input', type='number', value=0)
                ], className='mb-2'),
                
                dbc.InputGroup([
                    dbc.InputGroupText("Шаг"),
                    dbc.Input(id='gradient-lr-input', type='number', value=0.1, step=0.01)
                ], className='mb-2'),

                dbc.InputGroup([
                    dbc.InputGroupText("Точность ε (проверка убывания)"),
                    dbc.Input(id='gradient-epsilon-input', type='number', value=1e-4, step=1e-6)
                ], className='mb-2'),
                
                dbc.InputGroup([
                    dbc.InputGroupText("Точность ε1 (норма градиента в точке)"),
                    dbc.Input(id='gradient-epsilon1-input', type='number', value=1e-4, step=1e-6)
                ], className='mb-2'),

                dbc.InputGroup([
                    dbc.InputGroupText("Точность ε2 (разность значений функций)"),
                    dbc.Input(id='gradient-epsilon2-input', type='number', value=1e-4, step=1e-6)
                ], className='mb-2'),
                
                dbc.InputGroup([
                    dbc.InputGroupText("Макс. итераций"),
                    dbc.Input(id='gradient-max-iter-input', type='number', value=100)
                ], className='mb-2'),
                
                dbc.Button("Запустить", id='gradient-run-button', color='primary', className='mt-3')
            ])
    elif method_name == "Симплекс-метод":
        return html.Div([
            dbc.InputGroup([
                dbc.InputGroupText("x₁₀"),
                dbc.Input(id='simplex-x1-0-input', type='number', value=0)
            ], className='mb-2'),
            
            dbc.InputGroup([
                dbc.InputGroupText("x₂₀"),
                dbc.Input(id='simplex-x2-0-input', type='number', value=0)
            ], className='mb-2'),

            html.H4("Целевая функция F: a₁x₁² + a₂x₂² + a₃x₁x₂ + a₄x₁ + a₅x₂"),
            
            dbc.InputGroup([
                dbc.InputGroupText("Коэффициент​ x₁²"),
                dbc.Input(id='simplex-x1-2-input', type='number', value=0)
            ], className='mb-2'),

            dbc.InputGroup([
                dbc.InputGroupText("Коэффициент x₂²"),
                dbc.Input(id='simplex-x2-2-input', type='number', value=0)
            ], className='mb-2'),
            
            dbc.InputGroup([
                dbc.InputGroupText("Коэффициент x₁x₂"),
                dbc.Input(id='simplex-x1x2-input', type='number', value=0)
            ], className='mb-2'),

            dbc.InputGroup([
                dbc.InputGroupText("Коэффициент x₁"),
                dbc.Input(id='simplex-x1-input', type='number', value=0)
            ], className='mb-2'),
            
            dbc.InputGroup([
                dbc.InputGroupText("Коэффициент x₂"),
                dbc.Input(id='simplex-x2-input', type='number', value=0)
            ], className='mb-2'),
            
            html.H4("Условное ограничение 1: a₁x₁ + b₁x₂ ≤ c₁"),

            dbc.InputGroup([
                dbc.InputGroupText("Коэффициент a₁"),
                dbc.Input(id='simplex-a1-input', type='number', value=0)
            ], className='mb-2'),

            dbc.InputGroup([
                dbc.InputGroupText("Коэффициент b₁"),
                dbc.Input(id='simplex-b1-input', type='number', value=0)
            ], className='mb-2'),
            
            dbc.InputGroup([
                dbc.InputGroupText("Коэффициент c₁"),
                dbc.Input(id='simplex-c1-input', type='number', value=0)
            ], className='mb-2'),

            html.H4("Условное ограничение 2: a₂x₁ + b₂x₂ ≤ c₂"),

            dbc.InputGroup([
                dbc.InputGroupText("Коэффициент a₂"),
                dbc.Input(id='simplex-a2-input', type='number', value=0)
            ], className='mb-2'),

            dbc.InputGroup([
                dbc.InputGroupText("Коэффициент b₂"),
                dbc.Input(id='simplex-b2-input', type='number', value=0)
            ], className='mb-2'),
            
            dbc.InputGroup([
                dbc.InputGroupText("Коэффициент c₂"),
                dbc.Input(id='simplex-c2-input', type='number', value=0)
            ], className='mb-2'),

            dcc.Dropdown(
                id="simplex-type",
                options=["Минимум", "Максимум"],
                value="Минимум",
                clearable=False
            ),

            dbc.Button("Запустить", id='simplex-run-button', color='primary', className='mt-3')
        ])
    return html.Div()

@app.callback(
    [Output('3d-plot', 'figure', allow_duplicate=True),
     Output('results-table', 'children', allow_duplicate=True),
     Output('final-result', 'children', allow_duplicate=True),
     Output('final-result', 'color', allow_duplicate=True)],
    [Input('gradient-run-button', 'n_clicks')],
    [State('gradient-x0-input', 'value'),
     State('gradient-y0-input', 'value'),
     State('gradient-lr-input', 'value'),
     State('gradient-epsilon-input', 'value'),
     State('gradient-epsilon1-input', 'value'),
     State('gradient-epsilon2-input', 'value'),
     State('gradient-max-iter-input', 'value')],
     prevent_initial_call=True
)
def update_plot_and_table_gradient(n_clicks, x0, y0, lr, epsilon, epsilon1, epsilon2, max_iter):
    if None in [x0, y0, lr, epsilon, epsilon1, epsilon2, max_iter]:
        return go.Figure(), "Пожалуйста, заполните все поля", "", "danger"
    
    return update_plot_and_table("gradient", gradient_descent.func, *gradient_descent.optimize(x0, y0, lr, epsilon, epsilon1, epsilon2, max_iter))

@app.callback(
    [Output('3d-plot', 'figure', allow_duplicate=True),
     Output('results-table', 'children', allow_duplicate=True),
     Output('final-result', 'children', allow_duplicate=True),
     Output('final-result', 'color', allow_duplicate=True)],
    [Input('simplex-run-button', 'n_clicks')],
    [
     State('simplex-x1-0-input', 'value'),
     State('simplex-x2-0-input', 'value'),
     State('simplex-x1-2-input', 'value'),
     State('simplex-x2-2-input', 'value'),
     State('simplex-x1x2-input', 'value'),
     State('simplex-x1-input', 'value'),
     State('simplex-x2-input', 'value'),
     State('simplex-a1-input', 'value'),
     State('simplex-b1-input', 'value'),
     State('simplex-c1-input', 'value'),
     State('simplex-a2-input', 'value'),
     State('simplex-b2-input', 'value'),
     State('simplex-c2-input', 'value'),
     State('simplex-type', 'value')],
    prevent_initial_call=True
)
def update_plot_and_table_simplex(n_clicks, x0, y0, x12, x22, x1x2, x1, x2, a1, b1, c1, a2, b2, c2, type):
    if None in [x0, y0, x12, x22, x1x2, x1, x2, a1, b1, c1, a2, b2, c2, type]:
        return go.Figure(), "Пожалуйста, заполните все поля", "", "danger"
    
    return update_plot_and_table("simplex", simplex_method.objective_param([x12, x22, x1x2, x1, x2]), *simplex_method.optimize([x0, y0], [x12, x22, x1x2, x1, x2], [a1, b1, c1, a2, b2, c2], "minimize" if type=="Минимум" else "maximize"), [[a1, b1, c1, a2, b2, c2]])


def update_plot_and_table(method, func, history, converged, status_message, options=None):
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
    
    if method == "simplex":
        a1, b1, c1, a2, b2, c2 = options[0]
        
        func1 = lambda x1, x2: a1*x1 + b1*x2 - c1
        Z1 = func1(X, Y)

        func2 = lambda x1, x2: a2*x1 + b2*x2 - c2
        Z2 = func2(X, Y)
        
        fig.add_trace(go.Surface(x=X, y=Y, z=Z1, colorscale='Reds', opacity=0.2))
        fig.add_trace(go.Surface(x=X, y=Y, z=Z2, colorscale='Blues', opacity=0.2))

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
    ]

    if method == "gradient":
        columns.append({'name': 'Норма градиента', 'id': 'grad_norm'})
    
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