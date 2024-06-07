from dash import Dash, dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import numpy as np

#app = Dash(__name__)
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = Dash(__name__, external_stylesheets=external_stylesheets)

app.title = "Opgave 3"

def compute_wave_propagation(L, w, k0, v, N, dt, T):
    dx = L / N
    steps = int(T / dt)
    x = np.linspace(-L / 2, L / 2, N, endpoint=False)
    k = 2 * np.pi * np.fft.fftfreq(N, d=dx)
    # Initial condition
    u0 = np.cos(k0*x) * np.exp(-(x/w)**2)
    #np.exp(-(x**2/ w**2)) * (1/2*np.exp(-1j*k0*x) + 1/2*np.exp(1j*k0*x))

    # Numerical solution using FFT
    u_numerical = np.zeros((steps, N), dtype=np.complex64)
    A_numerical = np.zeros((steps, N), dtype=np.complex64)
    A_numerical[0, :] = np.fft.fft(u0)

    n_values = np.arange(1, steps)[:, np.newaxis]  # Add new axis to align shapes
    u_numerical[1:, :] = A_numerical[0,:]*np.cos(v*np.sqrt(k**2+20)*dt*n_values)

    u_numerical = np.fft.ifft(u_numerical, axis=1)

    return x, u_numerical, steps

# Set initial conditions
initial_L = 120
initial_w = 1
initial_k0 = 5
initial_N = 256
initial_dt = 0.025
initial_T = 15
initial_v = 1

# Perform initial computation
x, u_numerical, steps = compute_wave_propagation(initial_L, initial_w, initial_k0, initial_v, initial_N, initial_dt, initial_T)

markdown_text = r'''
## Opgave 3.


$
\frac{\partial^2 u}{\partial x^2} - \frac{1}{v^2} \frac{\partial^2 u}{\partial t^2} - u = 0
$
Med begyndelsesbetingelse: 

$
u(x, 0) = \exp\left(-\frac{x^2}{w^2}\right)\left(\frac{1}{2}e^{ik_0x} + \frac{1}{2}e^{-ik_0x}\right)
$
  og  
$
\frac{\partial u(x, 0)}{\partial t} = 0
$

'''


app.layout = html.Div([
    dcc.Markdown(children=markdown_text, mathjax=True, style={'fontSize':24}),

    html.Div([
        html.Div([
            html.Label('L:'),
            dcc.Input(id='L-value', type='number', value=initial_L, step=1, style={'width': '105%'})
        ], className='one column', style={'margin-right': '10px'}),
        
        html.Div([
            html.Label('w:'),
            dcc.Input(id='w-value', type='number', value=initial_w, step=1, style={'width': '105%'})
        ], className='one column', style={'margin-right': '10px'}),
        
        html.Div([
            html.Label('k0:'),
            dcc.Input(id='k0-value', type='number', value=initial_k0, step=1, style={'width': '105%'})
        ], className='one column', style={'margin-right': '10px'}),
        
        html.Div([
            html.Label('v:'),
            dcc.Input(id='v-value', type='number', value=initial_v, step=0.1, style={'width': '105%'})
        ], className='one column', style={'margin-right': '10px'}),
        
        html.Div([
            html.Label('N:'),
            dcc.Input(id='N-value', type='number', value=initial_N, step=1, style={'width': '105%'})
        ], className='one column', style={'margin-right': '10px'}),
        
        html.Div([
            html.Label('dt:'),
            dcc.Input(id='dt-value', type='number', value=initial_dt, step=0.001, style={'width': '105%'})
        ], className='one column', style={'margin-right': '10px'}),
        
        html.Div([
            html.Label('T:'),
            dcc.Input(id='T-value', type='number', value=initial_T, step=1, style={'width': '105%'})
        ], className='one column', style={'margin-right': '10px'}),

        html.Div([
            html.Button('Compute', id='compute-button', n_clicks=0, style={'margin-top': '10px'})
        ], className='one column', style={'margin-top': '22px'}),
    ], className='row', style={'margin-bottom': '20px'}),
    
    dcc.Graph(id='wave-animation'),  # This needs to be in the layout

    html.Div([
        html.Button('Play/Pause', id='play-pause-button', n_clicks=0, style={'margin-bottom': '10px'})
    ], style={'margin-bottom': '20px'}),
    
    html.Div([
        dcc.Slider(
            id='time-step-slider',
            min=0,
            max=steps - 1,
            value=0,
            step=1,
            marks={i: str(i) for i in range(0, steps, max(1, steps // 10))}
        )
    ], style={'margin-bottom': '20px'}),
    
    html.Div([
        dcc.Slider(
            id='speed-slider',
            min=200,
            max=1000,
            value=250,
            step=10,
            marks={i: f"{i}ms" for i in range(200, 1000, 100)}
        )
    ], style={'margin-bottom': '20px'}),
    
    dcc.Interval(
        id='interval-component',
        interval=1000, # in milliseconds
        n_intervals=0,
        disabled=True
    ),

    html.Div([
        dcc.Graph(
            id='3d-plot',
            figure={
                'data': [go.Surface(z=np.real(u_numerical))],
                'layout': go.Layout(
                    title='Løsning til Opgave 3',
                    scene=dict(
                        xaxis_title='x',
                        yaxis_title='t',
                        zaxis_title='u(x,t)',
                        xaxis=dict(tickvals=[], title='x'),  # Remove x-axis tick mark values
                        yaxis=dict(tickvals=[], title='t'),  # Remove y-axis tick mark values
                        zaxis=dict(tickvals=[], title='u(x,t)'),  # Remove z-axis tick mark values
                    ),
                    width=1000,
                    height=1000,
                    margin=dict(l=65, r=50, b=65, t=90),
                )
            }
        )
    ])
])




## Section 2
# Callbacks for updating the figure, playing/pausing, and animation speed

@app.callback(
    Output('wave-animation', 'figure'),
    [Input('time-step-slider', 'value'),
     Input('compute-button', 'n_clicks')],
    [State('L-value', 'value'),
     State('w-value', 'value'),
     State('k0-value', 'value'),
     State('v-value', 'value'),
     State('N-value', 'value'),
     State('dt-value', 'value'),
     State('T-value', 'value')]
)
def update_output(time_step, n_clicks, L, w, k0, v, N, dt, T):
    # Re-compute only if compute button is clicked
    ctx = callback_context
    if not ctx.triggered or ctx.triggered[0]['prop_id'].split('.')[0] == 'compute-button':
        global x, u_numerical, steps
        x, u_numerical, steps = compute_wave_propagation(L, w, k0, v, N, dt, T)
    fig = go.Figure(data=[
        go.Scatter(x=x, y=np.real(u_numerical[time_step, :]), mode='lines', name='Numerical FFT Solution'),
        
    ])
    fig.update_layout(
        title='Reeldel plottet',
        xaxis={'title': 'x'},
        yaxis={'title': 'Amplitude', 'range': [-1, 1]},
        margin={'l': 40, 'b': 40, 't': 50, 'r': 10},
        legend={'x': 0, 'y': 1},
        hovermode='closest'
    )
    return fig

@app.callback(
    [Output('interval-component', 'disabled'), Output('play-pause-button', 'children')],
    [Input('play-pause-button', 'n_clicks')],
    [State('interval-component', 'disabled')]
)
def toggle_play_pause(n_clicks, is_disabled):
    if is_disabled:
        return False, 'Pause'
    else:
        return True, 'Play'

@app.callback(
    Output('time-step-slider', 'value'),
    [Input('interval-component', 'n_intervals')],
    [State('time-step-slider', 'value'), State('time-step-slider', 'max')]
)
def advance_time_step(n_intervals, current_value, max_value):
    new_value = (current_value + 1) % (max_value + 1)
    return new_value

@app.callback(
    Output('interval-component', 'interval'),
    [Input('speed-slider', 'value')]
)
def update_speed(value):
    # Adjusting speed value conversion to interval if needed
    return max(200 - value, 10) * 10  # Example adjustment, modify as needed

# Update 'max' property of the time-step slider
@app.callback(
    Output('time-step-slider', 'max'),
    [Input('compute-button', 'n_clicks')],
    [State('T-value', 'value'), State('dt-value', 'value')]
)
def update_slider_max(n_clicks, T, dt):
    steps = int(T / dt)
    return steps - 1

from dash import Dash, dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import numpy as np

#app = Dash(__name__)
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = Dash(__name__, external_stylesheets=external_stylesheets)

app.title = "Opgave 3"

def compute_wave_propagation(L, w, k0, v, N, dt, T):
    dx = L / N
    steps = int(T / dt)
    x = np.linspace(-L / 2, L / 2, N, endpoint=False)
    k = 2 * np.pi * np.fft.fftfreq(N, d=dx)
    # Initial condition
    u0 = np.cos(k0*x) * np.exp(-(x/w)**2)
    #np.exp(-(x**2/ w**2)) * (1/2*np.exp(-1j*k0*x) + 1/2*np.exp(1j*k0*x))

    # Numerical solution using FFT
    u_numerical = np.zeros((steps, N), dtype=np.complex64)
    A_numerical = np.zeros((steps, N), dtype=np.complex64)
    A_numerical[0, :] = np.fft.fft(u0)

    n_values = np.arange(1, steps)[:, np.newaxis]  # Add new axis to align shapes
    u_numerical[1:, :] = A_numerical[0,:]*np.cos(v*np.sqrt(k**2+20)*dt*n_values)

    u_numerical = np.fft.ifft(u_numerical, axis=1)

    return x, u_numerical, steps

# Set initial conditions
initial_L = 120
initial_w = 1
initial_k0 = 5
initial_N = 256
initial_dt = 0.025
initial_T = 15
initial_v = 1

# Perform initial computation
x, u_numerical, steps = compute_wave_propagation(initial_L, initial_w, initial_k0, initial_v, initial_N, initial_dt, initial_T)

markdown_text = r'''
## Opgave 3.


$
\frac{\partial^2 u}{\partial x^2} - \frac{1}{v^2} \frac{\partial^2 u}{\partial t^2} - u = 0
$
Med begyndelsesbetingelse: 

$
u(x, 0) = \exp\left(-\frac{x^2}{w^2}\right)\left(\frac{1}{2}e^{ik_0x} + \frac{1}{2}e^{-ik_0x}\right)
$
  og  
$
\frac{\partial u(x, 0)}{\partial t} = 0
$

'''


app.layout = html.Div([
    dcc.Markdown(children=markdown_text, mathjax=True, style={'fontSize':24}),

    html.Div([
        html.Div([
            html.Label('L:'),
            dcc.Input(id='L-value', type='number', value=initial_L, step=1, style={'width': '105%'})
        ], className='one column', style={'margin-right': '10px'}),
        
        html.Div([
            html.Label('w:'),
            dcc.Input(id='w-value', type='number', value=initial_w, step=1, style={'width': '105%'})
        ], className='one column', style={'margin-right': '10px'}),
        
        html.Div([
            html.Label('k0:'),
            dcc.Input(id='k0-value', type='number', value=initial_k0, step=1, style={'width': '105%'})
        ], className='one column', style={'margin-right': '10px'}),
        
        html.Div([
            html.Label('v:'),
            dcc.Input(id='v-value', type='number', value=initial_v, step=0.1, style={'width': '105%'})
        ], className='one column', style={'margin-right': '10px'}),
        
        html.Div([
            html.Label('N:'),
            dcc.Input(id='N-value', type='number', value=initial_N, step=1, style={'width': '105%'})
        ], className='one column', style={'margin-right': '10px'}),
        
        html.Div([
            html.Label('dt:'),
            dcc.Input(id='dt-value', type='number', value=initial_dt, step=0.001, style={'width': '105%'})
        ], className='one column', style={'margin-right': '10px'}),
        
        html.Div([
            html.Label('T:'),
            dcc.Input(id='T-value', type='number', value=initial_T, step=1, style={'width': '105%'})
        ], className='one column', style={'margin-right': '10px'}),

        html.Div([
            html.Button('Compute', id='compute-button', n_clicks=0, style={'margin-top': '10px'})
        ], className='one column', style={'margin-top': '22px'}),
    ], className='row', style={'margin-bottom': '20px'}),
    
    dcc.Graph(id='wave-animation'),  # This needs to be in the layout

    html.Div([
        html.Button('Play/Pause', id='play-pause-button', n_clicks=0, style={'margin-bottom': '10px'})
    ], style={'margin-bottom': '20px'}),
    
    html.Div([
        dcc.Slider(
            id='time-step-slider',
            min=0,
            max=steps - 1,
            value=0,
            step=1,
            marks={i: str(i) for i in range(0, steps, max(1, steps // 10))}
        )
    ], style={'margin-bottom': '20px'}),
    
    html.Div([
        dcc.Slider(
            id='speed-slider',
            min=200,
            max=1000,
            value=250,
            step=10,
            marks={i: f"{i}ms" for i in range(200, 1000, 100)}
        )
    ], style={'margin-bottom': '20px'}),
    
    dcc.Interval(
        id='interval-component',
        interval=1000, # in milliseconds
        n_intervals=0,
        disabled=True
    ),

    html.Div([
        dcc.Graph(
            id='3d-plot',
            figure={
                'data': [go.Surface(z=np.real(u_numerical))],
                'layout': go.Layout(
                    title='Løsning til Opgave 3',
                    scene=dict(
                        xaxis_title='x',
                        yaxis_title='t',
                        zaxis_title='u(x,t)',
                        xaxis=dict(tickvals=[], title='x'),  # Remove x-axis tick mark values
                        yaxis=dict(tickvals=[], title='t'),  # Remove y-axis tick mark values
                        zaxis=dict(tickvals=[], title='u(x,t)'),  # Remove z-axis tick mark values
                    ),
                    width=1000,
                    height=1000,
                    margin=dict(l=65, r=50, b=65, t=90),
                )
            }
        )
    ])
])




## Section 2
# Callbacks for updating the figure, playing/pausing, and animation speed

@app.callback(
    Output('wave-animation', 'figure'),
    [Input('time-step-slider', 'value'),
     Input('compute-button', 'n_clicks')],
    [State('L-value', 'value'),
     State('w-value', 'value'),
     State('k0-value', 'value'),
     State('v-value', 'value'),
     State('N-value', 'value'),
     State('dt-value', 'value'),
     State('T-value', 'value')]
)
def update_output(time_step, n_clicks, L, w, k0, v, N, dt, T):
    # Re-compute only if compute button is clicked
    ctx = callback_context
    if not ctx.triggered or ctx.triggered[0]['prop_id'].split('.')[0] == 'compute-button':
        global x, u_numerical, steps
        x, u_numerical, steps = compute_wave_propagation(L, w, k0, v, N, dt, T)
    fig = go.Figure(data=[
        go.Scatter(x=x, y=np.real(u_numerical[time_step, :]), mode='lines', name='Numerical FFT Solution'),
        
    ])
    fig.update_layout(
        title='Reeldel plottet',
        xaxis={'title': 'x'},
        yaxis={'title': 'Amplitude', 'range': [-1, 1]},
        margin={'l': 40, 'b': 40, 't': 50, 'r': 10},
        legend={'x': 0, 'y': 1},
        hovermode='closest'
    )
    return fig

@app.callback(
    [Output('interval-component', 'disabled'), Output('play-pause-button', 'children')],
    [Input('play-pause-button', 'n_clicks')],
    [State('interval-component', 'disabled')]
)
def toggle_play_pause(n_clicks, is_disabled):
    if is_disabled:
        return False, 'Pause'
    else:
        return True, 'Play'

@app.callback(
    Output('time-step-slider', 'value'),
    [Input('interval-component', 'n_intervals')],
    [State('time-step-slider', 'value'), State('time-step-slider', 'max')]
)
def advance_time_step(n_intervals, current_value, max_value):
    new_value = (current_value + 1) % (max_value + 1)
    return new_value

@app.callback(
    Output('interval-component', 'interval'),
    [Input('speed-slider', 'value')]
)
def update_speed(value):
    # Adjusting speed value conversion to interval if needed
    return max(200 - value, 10) * 10  # Example adjustment, modify as needed

# Update 'max' property of the time-step slider
@app.callback(
    Output('time-step-slider', 'max'),
    [Input('compute-button', 'n_clicks')],
    [State('T-value', 'value'), State('dt-value', 'value')]
)
def update_slider_max(n_clicks, T, dt):
    steps = int(T / dt)
    return steps - 1

@app.callback(
    Output('3d-plot', 'figure'),
    [Input('compute-button', 'n_clicks')],
    [State('L-value', 'value'),
     State('w-value', 'value'),
     State('k0-value', 'value'),
     State('v-value', 'value'),
     State('N-value', 'value'),
     State('dt-value', 'value'),
     State('T-value', 'value')]
)
def update_3d_plot(n_clicks, L, w, k0, v, N, dt, T):
    # Re-compute only if compute button is clicked
    ctx = callback_context
    if not ctx.triggered or ctx.triggered[0]['prop_id'].split('.')[0] == 'compute-button':
        x, u_numerical, steps = compute_wave_propagation(L, w, k0, v, N, dt, T)
    fig_3d = {
        'data': [go.Surface(z=np.real(u_numerical))],
        'layout': go.Layout(
            title='Løsning til Lineære del af ikke-lineære Schrödingerligning',
            scene=dict(
                xaxis_title='x',
                yaxis_title='t',
                zaxis_title='u(x,t)',
                xaxis=dict(tickvals=[], title='x'),  # Remove x-axis tick mark values
                yaxis=dict(tickvals=[], title='t'),  # Remove y-axis tick mark values
                zaxis=dict(tickvals=[], title='u(x,t)'),  # Remove z-axis tick mark values
            ),
            width=1000,
            height=750,
            margin=dict(l=65, r=50, b=65, t=90),
        )
    }
    return fig_3d


if __name__ == '__main__':
    app.run_server(port=8053, debug=True)
