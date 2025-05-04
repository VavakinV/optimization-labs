import dash
from dash import dcc, html, Input, Output, State, dash_table
import plotly.graph_objs as go
import numpy as np
import dash_bootstrap_components as dbc

from methods import gradient_descent, simplex_method, genetic_algorithm, particle_swarm, bee, immune, bacterial
from functions import functions

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True, prevent_initial_callbacks='initial_duplicate')
server = app.server

methods = {
    "Градиентный спуск": gradient_descent,
    "Симплекс-метод": simplex_method,
    "Генетический алгоритм": genetic_algorithm,
    "Алгоритм роя частиц": particle_swarm,
    "Пчелиный алгоритм": bee,
    "Искусственная иммунная сеть": immune,
    "Бактериальная оптимизация": bacterial
}

optimization_functions = {
    "Функция Розенброка": "rosenbrock",
    "Функция Растригина": "rastrigin",
    "Функция Букина N6": "bukin",
    "Функция Химмельблау": "himmelblau",
    "Функция Изома": "isom",
    "Функция Гольдшейна-Прайса": "goldstein_price",
    "Функция 'Крест на подносе'": "cross_in_tray",
    "Функция сферы": "sphere"
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
    elif method_name == "Генетический алгоритм":
        return html.Div([
            dbc.InputGroup([
                dbc.InputGroupText("Функция"),
                dcc.Dropdown(
                        id="genetic-function-dropdown",
                        options=[{"label": name, "value": optimization_functions[name]} for name in optimization_functions.keys()],
                        value="rosenbrock",
                        clearable=False,
                        style={'width': '100%'}
                    )
            ], className='mb-2'),
            
            dbc.InputGroup([
                dbc.InputGroupText("Количество хромосом"),
                dbc.Input(id='genetic-chromosome-number-input', type='number', value=1000)
            ], className='mb-2'),

            dbc.InputGroup([
                dbc.InputGroupText("Макс. итераций"),
                dbc.Input(id='genetic-max-iter-input', type='number', value=25)
            ], className='mb-2'),

            dbc.InputGroup([
                dbc.InputGroupText("Интервал поиска (x)"),
                dbc.Input(id='genetic-x-min-input', type='number', value=-3, placeholder="Min x"),
                dbc.Input(id='genetic-x-max-input', type='number', value=3, placeholder="Max x")
            ], className='mb-2'),

            dbc.InputGroup([
                dbc.InputGroupText("Интервал поиска (y)"),
                dbc.Input(id='genetic-y-min-input', type='number', value=-3, placeholder="Min y"),
                dbc.Input(id='genetic-y-max-input', type='number', value=3, placeholder="Max y")
            ], className='mb-2'),

            dbc.InputGroup([
                dbc.InputGroupText("Вероятность скрещивания"),
                dbc.Input(id='genetic-crossover-prob-input', type='number', value=0.7)
            ], className='mb-2'),

            dbc.InputGroup([
                dbc.InputGroupText("Вероятность мутации"),
                dbc.Input(id='genetic-mutation-prob-input', type='number', value=0.1)
            ], className='mb-2'),

            dbc.InputGroup([
                dbc.InputGroupText("Параметр мутации"),
                dbc.Input(id='genetic-mutation-parameter-input', type='number', value=10)
            ], className='mb-2'),

            dbc.Checklist(
                options=[
                    {"label": "Использовать скрещивание", "value": "crossover"},
                    {"label": "Использовать мутацию", "value": "mutation"}
                ],
                value=["crossover", "mutation"],
                id="genetic-operations-checkboxes",
                switch=True,
                className='mb-2'
            ),

            dbc.Button("Запустить", id='genetic-run-button', color='primary', className='mt-3')
        ])
    elif method_name == "Алгоритм роя частиц":
        return html.Div([
            dbc.InputGroup([
                dbc.InputGroupText("Функция"),
                dcc.Dropdown(
                        id="swarm-function-dropdown",
                        options=[{"label": name, "value": optimization_functions[name]} for name in optimization_functions.keys()],
                        value="rosenbrock",
                        clearable=False,
                        style={'width': '100%'}
                    )
            ], className='mb-2'),
            
            dbc.InputGroup([
                dbc.InputGroupText("Размер роя"),
                dbc.Input(id='swarm-size-input', type='number', value=100)
            ], className='mb-2'),

            dbc.InputGroup([
                dbc.InputGroupText("Макс. итераций"),
                dbc.Input(id='swarm-max-iter-input', type='number', value=100)
            ], className='mb-2'),

            dbc.InputGroup([
                dbc.InputGroupText("Интервал поиска (x)"),
                dbc.Input(id='swarm-x-min-input', type='number', value=-3, placeholder="Min x"),
                dbc.Input(id='swarm-x-max-input', type='number', value=3, placeholder="Max x")
            ], className='mb-2'),

            dbc.InputGroup([
                dbc.InputGroupText("Интервал поиска (y)"),
                dbc.Input(id='swarm-y-min-input', type='number', value=-3, placeholder="Min y"),
                dbc.Input(id='swarm-y-max-input', type='number', value=3, placeholder="Max y")
            ], className='mb-2'),

            dbc.InputGroup([
                dbc.InputGroupText("Коэффициент k (должен быть в интервале (0, 1))"),
                dbc.Input(id='swarm-velocity-input', type='number', value=0.5)
            ], className='mb-2'),

            dbc.InputGroup([
                dbc.InputGroupText("Локальный коэффициент скорости φₚ"),
                dbc.Input(id='swarm-local-velocity-input', type='number', value=2)
            ], className='mb-2'),

            dbc.InputGroup([
                dbc.InputGroupText("Глобальный коэффициент скорости φ₉"),
                dbc.Input(id='swarm-global-velocity-input', type='number', value=5)
            ], className='mb-2'),

            dbc.InputGroup([
                dbc.InputGroupText("Коэффициент штрафа"),
                dbc.Input(id='swarm-penalty-input', type='number', value=10000)
            ], className='mb-2'),

            dbc.Button("Запустить", id='swarm-run-button', color='primary', className='mt-3')
        ])
    elif method_name == "Пчелиный алгоритм":
        return html.Div([
            dbc.InputGroup([
                dbc.InputGroupText("Функция"),
                dcc.Dropdown(
                        id="bee-function-dropdown",
                        options=[{"label": name, "value": optimization_functions[name]} for name in optimization_functions.keys()],
                        value="rosenbrock",
                        clearable=False,
                        style={'width': '100%'}
                    )
            ], className='mb-2'),

            dbc.InputGroup([
                dbc.InputGroupText("Макс. итераций"),
                dbc.Input(id='bee-max-iter-input', type='number', value=100)
            ], className='mb-2'),

            dbc.InputGroup([
                dbc.InputGroupText("Интервал поиска (x)"),
                dbc.Input(id='bee-x-min-input', type='number', value=-5, placeholder="Min x"),
                dbc.Input(id='bee-x-max-input', type='number', value=5, placeholder="Max x")
            ], className='mb-2'),

            dbc.InputGroup([
                dbc.InputGroupText("Интервал поиска (y)"),
                dbc.Input(id='bee-y-min-input', type='number', value=-5, placeholder="Min y"),
                dbc.Input(id='bee-y-max-input', type='number', value=5, placeholder="Max y")
            ], className='mb-2'),

            dbc.InputGroup([
                dbc.InputGroupText("Кол-во пчёл-разведчиков"),
                dbc.Input(id='bee-scoutbees-input', type='number', value=100)
            ], className='mb-2'),

            dbc.InputGroup([
                dbc.InputGroupText("Кол-во пчёл на элитных участках"),
                dbc.Input(id='bee-bestbees-input', type='number', value=50)
            ], className='mb-2'),

            dbc.InputGroup([
                dbc.InputGroupText("Кол-во пчёл на перспективных участках"),
                dbc.Input(id='bee-selbees-input', type='number', value=10)
            ], className='mb-2'),

            dbc.InputGroup([
                dbc.InputGroupText("Кол-во элитных участков"),
                dbc.Input(id='bee-bestsites-input', type='number', value=5)
            ], className='mb-2'),

            dbc.InputGroup([
                dbc.InputGroupText("Кол-во перспективных участков"),
                dbc.Input(id='bee-selsites-input', type='number', value=5)
            ], className='mb-2'),

            dbc.InputGroup([
                dbc.InputGroupText("Радиус поиска"),
                dbc.Input(id='bee-radius-input', type='number', value=0.5)
            ], className='mb-2'),

            dbc.InputGroup([
                dbc.InputGroupText("Коэффициент изменения участков"),
                dbc.Input(id='bee-koeff-input', type='number', value=0.2)
            ], className='mb-2'),

            dbc.InputGroup([
                dbc.InputGroupText("Итераций стагнации до расширения участка"),
                dbc.Input(id='bee-tolerance-input', type='number', value=5)
            ], className='mb-2'),

            dbc.InputGroup([
                dbc.InputGroupText("Макс. расширений участка"),
                dbc.Input(id='bee-globaltolerance-input', type='number', value=10)
            ], className='mb-2'),

            dbc.Button("Запустить", id='bee-run-button', color='primary', className='mt-3')
        ])
    elif method_name == "Искусственная иммунная сеть":
        return html.Div([
            dbc.InputGroup([
                dbc.InputGroupText("Функция"),
                dcc.Dropdown(
                        id="immune-function-dropdown",
                        options=[{"label": name, "value": optimization_functions[name]} for name in optimization_functions.keys()],
                        value="rosenbrock",
                        clearable=False,
                        style={'width': '100%'}
                    )
            ], className='mb-2'),

            dbc.InputGroup([
                dbc.InputGroupText("Макс. итераций"),
                dbc.Input(id='immune-max-iter-input', type='number', value=10000)
            ], className='mb-2'),

            dbc.InputGroup([
                dbc.InputGroupText("Кол-во антител"),
                dbc.Input(id='immune-population-input', type='number', value=50)
            ], className='mb-2'),

            dbc.InputGroup([
                dbc.InputGroupText("Интервал поиска (x)"),
                dbc.Input(id='immune-x-min-input', type='number', value=-5, placeholder="Min x"),
                dbc.Input(id='immune-x-max-input', type='number', value=5, placeholder="Max x")
            ], className='mb-2'),

            dbc.InputGroup([
                dbc.InputGroupText("Интервал поиска (y)"),
                dbc.Input(id='immune-y-min-input', type='number', value=-5, placeholder="Min y"),
                dbc.Input(id='immune-y-max-input', type='number', value=5, placeholder="Max y")
            ], className='mb-2'),

            dbc.InputGroup([
                dbc.InputGroupText("Кол-во лучших антител"),
                dbc.Input(id='immune-nb-input', type='number', value=10)
            ], className='mb-2'),

            dbc.InputGroup([
                dbc.InputGroupText("Кол-во клонов каждого лучшего антитела"),
                dbc.Input(id='immune-nc-input', type='number', value=3)
            ], className='mb-2'),

            dbc.InputGroup([
                dbc.InputGroupText("Кол-во лучших клонов, оставляемых после мутации"),
                dbc.Input(id='immune-nd-input', type='number', value=5)
            ], className='mb-2'),

            dbc.InputGroup([
                dbc.InputGroupText("Коэффициент мутации"),
                dbc.Input(id='immune-mutation-input', type='number', value=0.1)
            ], className='mb-2'),

            dbc.InputGroup([
                dbc.InputGroupText("Макс. итераций без улучшения"),
                dbc.Input(id='immune-tolerance-steps-input', type='number', value=100)
            ], className='mb-2'),

            dbc.Button("Запустить", id='immune-run-button', color='primary', className='mt-3')
        ])
    elif method_name == "Бактериальная оптимизация":
        return html.Div([
            dbc.InputGroup([
                dbc.InputGroupText("Функция"),
                dcc.Dropdown(
                        id="bacterial-function-dropdown",
                        options=[{"label": name, "value": optimization_functions[name]} for name in optimization_functions.keys()],
                        value="sphere",
                        clearable=False,
                        style={'width': '100%'}
                    )
            ], className='mb-2'),

            dbc.InputGroup([
                dbc.InputGroupText("Макс. итераций (кол-во хемотаксисов)"),
                dbc.Input(id='bacterial-chemotaxis-input', type='number', value=100)
            ], className='mb-2'),

            dbc.InputGroup([
                dbc.InputGroupText("Кол-во бактерий"),
                dbc.Input(id='bacterial-population-input', type='number', value=100)
            ], className='mb-2'),

            dbc.InputGroup([
                dbc.InputGroupText("Интервал поиска (x)"),
                dbc.Input(id='bacterial-x-min-input', type='number', value=-5, placeholder="Min x"),
                dbc.Input(id='bacterial-x-max-input', type='number', value=5, placeholder="Max x")
            ], className='mb-2'),

            dbc.InputGroup([
                dbc.InputGroupText("Интервал поиска (y)"),
                dbc.Input(id='bacterial-y-min-input', type='number', value=-5, placeholder="Min y"),
                dbc.Input(id='bacterial-y-max-input', type='number', value=5, placeholder="Max y")
            ], className='mb-2'),

            dbc.InputGroup([
                dbc.InputGroupText("Кол-во репродукций"),
                dbc.Input(id='bacterial-reproductions-input', type='number', value=100)
            ], className='mb-2'),

            dbc.InputGroup([
                dbc.InputGroupText("Кол-во ликвидаций и рассеиваний"),
                dbc.Input(id='bacterial-eliminations-input', type='number', value=50)
            ], className='mb-2'),

            dbc.InputGroup([
                dbc.InputGroupText("Шаг хемотаксиса"),
                dbc.Input(id='bacterial-chemstep-input', type='number', value=0.1)
            ], className='mb-2'),

            dbc.InputGroup([
                dbc.InputGroupText("Порог шагов до ликвидации"),
                dbc.Input(id='bacterial-threshold-input', type='number', value=30)
            ], className='mb-2'),

            dbc.InputGroup([
                dbc.InputGroupText("Вероятность ликвидации"),
                dbc.Input(id='bacterial-elimination-probability-input', type='number', value=0.25)
            ], className='mb-2'),

            dbc.InputGroup([
                dbc.InputGroupText("Кол-во ликвидируемых бактерий"),
                dbc.Input(id='bacterial-elimination-count-input', type='number', value=3)
            ], className='mb-2'),

            dbc.Button("Запустить", id='bacterial-run-button', color='primary', className='mt-3')
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
    
    return update_plot_and_table("simplex", simplex_method.objective_param([x12, x22, x1x2, x1, x2]), *simplex_method.optimize([x0, y0], [x12, x22, x1x2, x1, x2], [a1, b1, c1, a2, b2, c2] if (any([x != 0 for x in [a1, b1, c1]]) and any([x != 0 for x in [a2, b2, c2]])) else [a1, b1, c1] if (any([x != 0 for x in [a1, b1, c1]]) and all([x == 0 for x in [a2, b2, c2]])) else [a2, b2, c2] if (all([x == 0 for x in [a1, b1, c1]]) and any([x != 0 for x in [a2, b2, c2]])) else [], "minimize" if type=="Минимум" else "maximize"), {'simplex_coefficients': [a1, b1, c1, a2, b2, c2]})

@app.callback(
    [Output('3d-plot', 'figure', allow_duplicate=True),
     Output('results-table', 'children', allow_duplicate=True),
     Output('final-result', 'children', allow_duplicate=True),
     Output('final-result', 'color', allow_duplicate=True)],
    [Input('genetic-run-button', 'n_clicks')],
    [
     State('genetic-function-dropdown', 'value'),
     State('genetic-chromosome-number-input', 'value'),
     State('genetic-max-iter-input', 'value'),
     State('genetic-x-min-input', 'value'),
     State('genetic-x-max-input', 'value'),
     State('genetic-y-min-input', 'value'),
     State('genetic-y-max-input', 'value'),
     State('genetic-crossover-prob-input', 'value'),
     State('genetic-mutation-prob-input', 'value'),
     State('genetic-mutation-parameter-input', 'value'),
     State('genetic-operations-checkboxes', 'value'),
     ],
    prevent_initial_call=True
)
def update_plot_and_table_genetic(n_clicks, func, chromosome_number, max_iter, x_min, x_max, y_min, y_max, crossover_prob, mutation_prob, mutation_param, operations):
    func = functions(func)
    if None in [func, chromosome_number, max_iter, x_min, x_max, y_min, y_max, crossover_prob, mutation_prob, mutation_param, operations]:
        return go.Figure(), "Пожалуйста, заполните все поля", "", "danger"

    return update_plot_and_table("genetic", func, *genetic_algorithm.optimize(func, [[x_min, x_max], [y_min, y_max]], {"crossover": "crossover" in operations, "mutation": "mutation" in operations}, chromosome_number, crossover_prob, mutation_prob, mutation_param, max_iter), {"bounds": [x_min, x_max, y_min, y_max]})


@app.callback(
    [Output('3d-plot', 'figure', allow_duplicate=True),
     Output('results-table', 'children', allow_duplicate=True),
     Output('final-result', 'children', allow_duplicate=True),
     Output('final-result', 'color', allow_duplicate=True)],
    [Input('swarm-run-button', 'n_clicks')],
    [
     State('swarm-function-dropdown', 'value'),
     State('swarm-size-input', 'value'),
     State('swarm-max-iter-input', 'value'),
     State('swarm-x-min-input', 'value'),
     State('swarm-x-max-input', 'value'),
     State('swarm-y-min-input', 'value'),
     State('swarm-y-max-input', 'value'),
     State('swarm-velocity-input', 'value'),
     State('swarm-local-velocity-input', 'value'),
     State('swarm-global-velocity-input', 'value'),
     State('swarm-penalty-input', 'value')
     ],
    prevent_initial_call=True
)
def update_plot_and_table_swarm(n_clicks, func, swarmsize, max_iter, x_min, x_max, y_min, y_max, velocity, local_velocity, global_velocity, penalty):
    func = functions(func)
    if None in [func, swarmsize, max_iter, x_min, x_max, y_min, y_max, velocity, local_velocity, global_velocity, penalty]:
        return go.Figure(), "Пожалуйста, заполните все поля", "", "danger"
    
    if 0 >= velocity or velocity >= 1:
        return go.Figure(), "Коэффициент k должен быть в диапазоне (0, 1)", "", "danger"

    return update_plot_and_table("swarm", func, *particle_swarm.optimize(func, max_iter, swarmsize, [[x_min, y_min], [x_max, y_max]], velocity, local_velocity, global_velocity, penalty), {"bounds": [x_min, x_max, y_min, y_max]})

@app.callback(
    [Output('3d-plot', 'figure', allow_duplicate=True),
     Output('results-table', 'children', allow_duplicate=True),
     Output('final-result', 'children', allow_duplicate=True),
     Output('final-result', 'color', allow_duplicate=True)],
    [Input('bee-run-button', 'n_clicks')],
    [
     State('bee-function-dropdown', 'value'),
     State('bee-max-iter-input', 'value'),
     State('bee-x-min-input', 'value'),
     State('bee-x-max-input', 'value'),
     State('bee-y-min-input', 'value'),
     State('bee-y-max-input', 'value'),
     State('bee-scoutbees-input', 'value'),
     State('bee-bestbees-input', 'value'),
     State('bee-selbees-input', 'value'),
     State('bee-bestsites-input', 'value'),
     State('bee-selsites-input', 'value'),
     State('bee-radius-input', 'value'),
     State('bee-koeff-input', 'value'),
     State('bee-tolerance-input', 'value'),
     State('bee-globaltolerance-input', 'value'),
     ],
    prevent_initial_call=True
)
def update_plot_and_table_bee(n_clicks, func, max_iter, x_min, x_max, y_min, y_max, scoutbees, bestbees, selbees, bestsites, selsites, radius, koeff, tolerance, globaltolerance):
    func = functions(func)
    if None in [func, max_iter, x_min, x_max, y_min, y_max, scoutbees, bestbees, selbees, bestsites, selsites, radius, koeff, tolerance, globaltolerance]:
        return go.Figure(), "Пожалуйста, заполните все поля", "", "danger"
    
    if 0 >= koeff or koeff > 1:
        return go.Figure(), "Коэффициент изменения участков должен быть в диапазоне (0, 1]", "", "danger"

    return update_plot_and_table("bee", func, *bee.optimize(func, max_iter, scoutbees, selbees, bestbees, bestsites, selsites, radius, koeff, tolerance, globaltolerance, x_min, x_max, y_min, y_max), {"bounds": [x_min, x_max, y_min, y_max]})

@app.callback(
    [Output('3d-plot', 'figure', allow_duplicate=True),
     Output('results-table', 'children', allow_duplicate=True),
     Output('final-result', 'children', allow_duplicate=True),
     Output('final-result', 'color', allow_duplicate=True)],
    [Input('immune-run-button', 'n_clicks')],
    [
     State('immune-function-dropdown', 'value'),
     State('immune-max-iter-input', 'value'),
     State('immune-population-input', 'value'),
     State('immune-x-min-input', 'value'),
     State('immune-x-max-input', 'value'),
     State('immune-y-min-input', 'value'),
     State('immune-y-max-input', 'value'),
     State('immune-nb-input', 'value'),
     State('immune-nc-input', 'value'),
     State('immune-nd-input', 'value'),
     State('immune-mutation-input', 'value'),
     State('immune-tolerance-steps-input', 'value'),
     ],
    prevent_initial_call=True
)
def update_plot_and_table_immune(n_clicks, func, max_iter, population, x_min, x_max, y_min, y_max, nb, nc, nd, mutation, tolerance_steps):
    func = functions(func)
    if None in [func, max_iter, population, x_min, x_max, y_min, y_max, nb, nc, nd, mutation, tolerance_steps]:
        return go.Figure(), "Пожалуйста, заполните все поля", "", "danger"

    return update_plot_and_table("immune", func, *immune.optimize(func, max_iter, population, x_min, x_max, y_min, y_max, nb, nc, nd, mutation, tolerance_steps), {"bounds": [x_min, x_max, y_min, y_max]})

@app.callback(
    [Output('3d-plot', 'figure', allow_duplicate=True),
     Output('results-table', 'children', allow_duplicate=True),
     Output('final-result', 'children', allow_duplicate=True),
     Output('final-result', 'color', allow_duplicate=True)],
    [Input('bacterial-run-button', 'n_clicks')],
    [
     State('bacterial-function-dropdown', 'value'),
     State('bacterial-x-min-input', 'value'),
     State('bacterial-x-max-input', 'value'),
     State('bacterial-y-min-input', 'value'),
     State('bacterial-y-max-input', 'value'),
     State('bacterial-population-input', 'value'),
     State('bacterial-chemotaxis-input', 'value'),
     State('bacterial-reproductions-input', 'value'),
     State('bacterial-eliminations-input', 'value'),
     State('bacterial-chemstep-input', 'value'),
     State('bacterial-threshold-input', 'value'),
     State('bacterial-elimination-probability-input', 'value'),
     State('bacterial-elimination-count-input', 'value'),
     ],
    prevent_initial_call=True
)
def update_plot_and_table_immune(n_clicks, func, x_min, x_max, y_min, y_max, population_count, n_chemotaxis, n_reproduction, n_elimination, chemotaxis_step, elimination_threshold, elimination_probabilty, elimination_count):
    func = functions(func)
    if None in [func, x_min, x_max, y_min, y_max, population_count, n_chemotaxis, n_reproduction, n_elimination, chemotaxis_step, elimination_threshold, elimination_probabilty, elimination_count]:
        return go.Figure(), "Пожалуйста, заполните все поля", "", "danger"
    
    if (population_count % 2) == 1:
        return go.Figure(), "Количество бактерий в популяции должно быть чётным", "", "danger"

    if (n_reproduction > n_chemotaxis) or (n_elimination > n_chemotaxis):
        return go.Figure(), "Количество репродукций и ликвидаций должно быть не больше общего кол-ва итераций", "", "danger"
    
    if (elimination_probabilty < 0) or (elimination_probabilty > 1):
        return go.Figure(), "Вероятность ликвидации должна быть в диапазоне [0;1]", "", "danger"
    
    if elimination_count > population_count:
        return go.Figure(), "Кол-во ликвидируемых бактерий не должно превышать общий размер популяции", "", "danger"

    return update_plot_and_table("bacterial", func, *bacterial.optimize(func, x_min, x_max, y_min, y_max, population_count, n_chemotaxis, n_reproduction, n_elimination, chemotaxis_step, elimination_threshold, elimination_probabilty, elimination_count), {"bounds": [x_min, x_max, y_min, y_max]})


def update_plot_and_table(method, func, history, converged, status_message, options=None, optional_options=None):
    if history:
        final = history[-1]
        result_message = [
            html.Strong("Результаты оптимизации:"),
            html.Br(),
            f"Финальная точка: ({round(final['x'], 5)}, {round(final['y'], 5)})",
            html.Br(),
            f"Значение функции: {round(final['f_value'], 5)}",
            html.Br(),
            f"Итераций выполнено: {final['iteration']}",
            html.Br(),
            f"Состояние: {status_message}"
        ]
        color = "success" if converged else "warning"
    else:
        result_message = "Не удалось выполнить оптимизацию"
        color = "danger"
    
    x = np.linspace(0, 20, 100)
    y = np.linspace(0, 20, 100)
    if method in ['genetic', 'swarm', 'bee', 'immune', 'bacterial']:
        x_min, x_max, y_min, y_max = options["bounds"]
        x = np.linspace(x_min, x_max, 100)
        y = np.linspace(y_min, y_max, 100)
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
        a1, b1, c1, a2, b2, c2 = options['simplex_coefficients']
        
        if any([x !=0 for x in [a1, b1, c1]]):
            func1 = lambda x1, x2: a1*x1 + b1*x2 - c1
            Z1 = func1(X, Y)
            fig.add_trace(go.Surface(x=X, y=Y, z=Z1, colorscale='Reds', opacity=0.4))

        if any([x != 0 for x in [a2, b2, c2]]):
            func2 = lambda x1, x2: a2*x1 + b2*x2 - c2
            Z2 = func2(X, Y)
            fig.add_trace(go.Surface(x=X, y=Y, z=Z2, colorscale='Blues', opacity=0.4))

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

    formatted_history = [
        {
            'iteration': item['iteration'],
            'x': round(item['x'], 5),
            'y': round(item['y'], 5),
            'f_value': round(item['f_value'], 5),
            
        }
        for item in history
    ]
    
    if method == 'gradient':
        columns.append({'name': 'Норма градиента', 'id': 'grad_norm'})
        formatted_optional_history = [{
                'grad_norm': round(item['grad_norm'], 5)
            }
            for item in options['optional_history']
        ]
        for i in range(len(formatted_history)):
            formatted_history[i]['grad_norm'] = formatted_optional_history[i]['grad_norm']

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