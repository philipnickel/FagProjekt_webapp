import time

import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objs as go
from dash import Dash, Input, Output, State, callback_context, dcc, html

external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]
app = Dash(__name__, external_stylesheets=external_stylesheets)

app.title = "SSFM"


# Full step split step fourier method
def split_step_fourier_full_step(C1, C2, u, k, dt):
    # Nonlinear part
    u *= np.exp((C2 * 1j * (np.abs(u) ** 2) * dt))
    # Linear part
    u_fft = np.fft.fft(u)
    u_fft *= np.exp(C1 * -1j * (k**2) * dt)
    u = np.fft.ifft(u_fft)

    return u


# Half step split step fourier method
def split_step_fourier(C1, C2, u, k, dt):
    # Nonlinear part
    u *= np.exp((C2 * 1j * (np.abs(u) ** 2) * dt / 2))

    # Linear part
    u_fft = np.fft.fft(u)
    u_fft *= np.exp(C1 * -1j * (k**2) * dt)
    u = np.fft.ifft(u_fft)

    # Nonlinear part, again
    u *= np.exp((C2 * 1j * (np.abs(u) ** 2) * dt / 2))

    return u


def compute_wave_propagation(
    C1, C2, L, sigma, v, theta, N, dt, T, initial_condition, method
):
    steps = int(T / dt)
    # Spatial domain setup
    dx = L / N
    x = np.linspace(-L / 2, L / 2, N, endpoint=False)
    k = 2 * np.pi * np.fft.fftfreq(N, d=dx)

    A = 1  # Amplitude

    if initial_condition == 1:
        # Initial condition (Gaussian wave packet)
        # u0 = np.exp(-(x/w)**2 + 1j * k0 * x)
        u0 = 0
    elif initial_condition == 2:
        u0 = (
            A
            * 1
            / np.cosh(A * np.sqrt(C2 / (2 * C1)) * (x - theta))
            * np.exp(1j * v / (2 * C1) * x + 1j * sigma)
        )

    # Prepare the storage for the solution at each time step
    u_storage = np.zeros((steps, N), dtype=np.complex64)
    u = u0

    u_analytical = np.zeros((steps, N), dtype=np.complex64)

    if method == "half":
        for i in range(steps):
            # Perform SSFM step
            u_storage[i, :] = u
            u = split_step_fourier(C1, C2, u, k, dt)
    elif method == "full":
        for i in range(steps):
            # Perform SSFM step
            u_storage[i, :] = u
            u = split_step_fourier_full_step(C1, C2, u, k, dt)

    x = np.linspace(-L / 2, L / 2, N, endpoint=False)
    n_values = np.arange(1, steps)[:, np.newaxis]  # Add new axis to align shapes

    u_analytical[0, :] = (
        A
        * 1
        / np.cosh(A * np.sqrt(C2 / (2 * C1)) * (x - theta))
        * np.exp(1j * v / (2 * C1) * x + 1j * sigma)
    )
    u_analytical[1:, :] = (
        A
        * 1
        / np.cosh(A * np.sqrt(C2 / (2 * C1)) * (x - v * dt * n_values - theta))
        * np.exp(
            1j * v / (2 * C1) * x
            + 1j * ((C2 * A**2) / 2 - (v**2 / (4 * C1))) * dt * n_values
            + 1j * sigma
        )
    )

    return x, u_storage, u_analytical, steps


# Set initial conditions
initial_L = 30
initial_T = 10
initial_sigma = 0
initial_N = 1000
initial_dt = 0.005
initial_v = 10
initial_condition = 2
initial_C1 = 1
initial_C2 = 2
initial_theta = 0
initial_interval = 10
initial_N2 = 500
initial_T2 = 10

# Perform initial computation
x, u_numerical, u_analytical, steps = compute_wave_propagation(
    initial_C1,
    initial_C2,
    initial_L,
    initial_sigma,
    initial_v,
    initial_theta,
    initial_N,
    initial_dt,
    initial_T,
    initial_condition,
    "half",
)

# markdown text before animation
markdown_text = r"""
## Split step Fourier metode (SSFM) for den ikke-lineære Schrödingerligning

$$
i \frac{\partial u}{\partial t} + C1 \frac{\partial^2 u}{\partial x^2}  + C2 |u|^2 u = 0
$$
Med en af følgende begyndelsesbetingelser: 

$$
u(x,0) = A \space sech(A \sqrt{\frac{C2}{2C1}}(x-\theta)) e^{i \frac{v}{2 \space C1}x + i \sigma}
$$

**_Bemærk_**: Amplituden $A$ er sat til 1.


"""
# Markdown text after 3d plot before precision/N plot and precision/cpu time plot
markdown_text_2 = r"""
## Sammenligning af præcision ved 'full step' og 'half step' split step Fourier metoderne 

"""


app.layout = html.Div(
    [
        dcc.Markdown(children=markdown_text, mathjax=True, style={"fontSize": 24}),
        html.Div(
            [
                html.Div(
                    [
                        html.Label("C1:"),
                        dcc.Input(
                            id="C1-value",
                            type="number",
                            value=initial_C1,
                            step=0.1,
                            style={"width": "110%"},
                        ),
                    ],
                    className="one column",
                    style={"margin-right": "2px"},
                ),
                html.Div(
                    [
                        html.Label("C2:"),
                        dcc.Input(
                            id="C2-value",
                            type="number",
                            value=initial_C2,
                            step=0.1,
                            style={"width": "110%"},
                        ),
                    ],
                    className="one column",
                    style={"margin-right": "2px"},
                ),
                html.Div(
                    [
                        html.Label("L:"),
                        dcc.Input(
                            id="L-value",
                            type="number",
                            value=initial_L,
                            step=1,
                            style={"width": "110%"},
                        ),
                    ],
                    className="one column",
                    style={"margin-right": "2px"},
                ),
                html.Div(
                    [
                        html.Label("sigma:"),
                        dcc.Input(
                            id="sigma-value",
                            type="number",
                            value=initial_sigma,
                            step=1,
                            style={"width": "110%"},
                        ),
                    ],
                    className="one column",
                    style={"margin-right": "2px"},
                ),
                html.Div(
                    [
                        html.Label("v:"),
                        dcc.Input(
                            id="v-value",
                            type="number",
                            value=initial_v,
                            step=0.1,
                            style={"width": "110%"},
                        ),
                    ],
                    className="one column",
                    style={"margin-right": "2px"},
                ),
                html.Div(
                    [
                        html.Label("theta:"),
                        dcc.Input(
                            id="theta-value",
                            type="number",
                            value=initial_theta,
                            step=0.1,
                            style={"width": "110%"},
                        ),
                    ],
                    className="one column",
                    style={"margin-right": "2px"},
                ),
                html.Div(
                    [
                        html.Label("N:"),
                        dcc.Input(
                            id="N-value",
                            type="number",
                            value=initial_N,
                            step=1,
                            style={"width": "110%"},
                        ),
                    ],
                    className="one column",
                    style={"margin-right": "2px"},
                ),
                html.Div(
                    [
                        html.Label("dt:"),
                        dcc.Input(
                            id="dt-value",
                            type="number",
                            value=initial_dt,
                            step=0.001,
                            style={"width": "110%"},
                        ),
                    ],
                    className="one column",
                    style={"margin-right": "2px"},
                ),
                html.Div(
                    [
                        html.Label("T:"),
                        dcc.Input(
                            id="T-value",
                            type="number",
                            value=initial_T,
                            step=1,
                            style={"width": "110%"},
                        ),
                    ],
                    className="one column",
                    style={"margin-right": "2px"},
                ),
                html.Div(
                    [
                        html.Label("Part:"),
                        dcc.Dropdown(
                            id="Part",
                            options=[
                                {"label": "Real", "value": 1},
                                {"label": "Imaginary", "value": 0},
                            ],
                            value=1,
                            style={"width": "130%"},
                        ),
                    ],
                    className="one column",
                    style={"margin-right": "2px"},
                ),
                html.Div(
                    [
                        html.Label("u0:"),
                        dcc.Dropdown(
                            id="initial-condition",
                            options=[
                                # {'label': 'Gaussian wave packet (1)', 'value': 1},
                                {"label": "Analytical Solution (2)", "value": 2},
                                # {'label': 'Cosine wave packet (2)', 'value': 2},
                                # {'label': 'Sech wave packet (3)', 'value': 3},
                                # {'label': 'Gaussian non-moving (4)', 'value': 4},
                                # {'label': 'Ny betingelse? 1', 'value': 6},
                                # {'label': 'Ny betingelse? 2', 'value': 7}
                            ],
                            value=initial_condition,
                            style={"width": "250px"},  # Adjust the width as needed
                        ),
                    ],
                    className="one column",
                    style={"margin-right": "2px"},
                ),
            ],
            className="row",
            style={"margin-bottom": "22px"},
        ),
        html.Div(
            [
                html.Button(
                    "Compute",
                    id="compute-button",
                    n_clicks=0,
                    style={"margin-top": "12px"},
                )
            ],
            className="row",
            style={"margin-bottom": "22px"},
        ),
        dcc.Graph(
            id="wave-animation", style={"height": "600px"}
        ),  # This needs to be in the layout
        html.Div(
            [
                html.Button(
                    "Play/Pause",
                    id="play-pause-button",
                    n_clicks=1,
                    style={"margin-bottom": "10px"},
                )
            ],
            style={"margin-bottom": "20px"},
        ),
        html.Div(
            [
                dcc.Slider(
                    id="time-step-slider",
                    min=0,
                    max=steps - 1,
                    value=0,
                    step=1,
                    marks={i: str(i) for i in range(0, steps, max(1, steps // 10))},
                )
            ],
            style={"margin-bottom": "20px"},
        ),
        html.Div(
            [
                dcc.Slider(
                    id="speed-slider",
                    min=100,
                    max=1000,
                    value=250,
                    step=10,
                    marks={i: f"{i}ms" for i in range(100, 1000, 100)},
                )
            ],
            style={"margin-bottom": "20px"},
        ),
        dcc.Interval(
            id="interval-component",
            interval=1000,  # in milliseconds
            n_intervals=0,
            disabled=True,
        ),
        html.Div(
            [
                dcc.Graph(
                    id="3d-plot",
                    figure={
                        "data": [go.Surface(z=np.real(u_analytical))],
                        "layout": go.Layout(
                            title="Løsning til den ikke-lineære Schrödingerligning",
                            scene=dict(
                                xaxis_title="x",
                                yaxis_title="t",
                                zaxis_title="u(x,t)",
                                xaxis=dict(
                                    tickvals=[], title="x"
                                ),  # Remove x-axis tick mark values
                                yaxis=dict(
                                    tickvals=[], title="t"
                                ),  # Remove y-axis tick mark values
                                zaxis=dict(
                                    tickvals=[], title="u(x,t)"
                                ),  # Remove z-axis tick mark values
                            ),
                            width=1000,
                            height=750,
                            margin=dict(l=65, r=50, b=65, t=90),
                        ),
                    },
                )
            ]
        ),
        dcc.Markdown(children=markdown_text_2, mathjax=True, style={"fontSize": 24}),
        # Input for selecting the number of points
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.Label("N:"),
                        dcc.Input(
                            id="N-value-2", type="number", value=initial_N2, step=1
                        ),
                    ],
                    width=3,
                ),
                dbc.Col(
                    [
                        html.Label("Interval:"),
                        dcc.Input(
                            id="Interval-value-2",
                            type="number",
                            value=initial_interval,
                            step=1,
                        ),
                    ],
                    width=3,
                ),
                dbc.Col(
                    [
                        html.Label("T:"),
                        dcc.Input(
                            id="T-value-2", type="number", value=initial_T2, step=1
                        ),
                    ],
                    width=3,
                ),
            ],
            style={"margin-bottom": "20px"},
        ),
        # Button for computing the solution
        html.Div(
            [
                html.Button(
                    "Compute",
                    id="compute-button-2",
                    n_clicks=0,
                    style={"margin-top": "12px"},
                )
            ],
            style={"margin-bottom": "22px"},
        ),
        html.Div(
            [
                dcc.Loading(
                    id="loading-precision-n",
                    type="graph",
                    children=[
                        dcc.Graph(
                            id="precision-n-plot",
                            style={"width": "100%", "height": "auto"},
                        )
                    ],
                )
            ],
            style={"margin-bottom": "20px"},
        ),
        html.Div(
            [
                dcc.Loading(
                    id="loading-precision-cpu-time",
                    type="graph",
                    children=[
                        dcc.Graph(
                            id="precision-cpu-time-plot",
                            style={"width": "100%", "height": "auto"},
                        )
                    ],
                )
            ],
            style={"margin-bottom": "20px"},
        ),
    ]
)


## Section 2
# Callbacks for updating the figure, playing/pausing, and animation speed


@app.callback(
    Output("wave-animation", "figure"),
    [Input("time-step-slider", "value"), Input("compute-button", "n_clicks")],
    [
        State("C1-value", "value"),
        State("C2-value", "value"),
        State("L-value", "value"),
        State("sigma-value", "value"),
        State("v-value", "value"),
        State("theta-value", "value"),
        State("N-value", "value"),
        State("dt-value", "value"),
        State("T-value", "value"),
        State("initial-condition", "value"),
        State("Part", "value"),
    ],
)
def update_line_plot(
    time_step, n_clicks, C1, C2, L, sigma, v, theta, N, dt, T, initial_condition, Part
):
    # Re-compute only if compute button is clicked
    ctx = callback_context
    if (
        not ctx.triggered
        or ctx.triggered[0]["prop_id"].split(".")[0] == "compute-button"
    ):
        global x, u_numerical, u_analytical, steps
        x, u_numerical, u_analytical, steps = compute_wave_propagation(
            C1, C2, L, sigma, v, theta, N, dt, T, initial_condition, "half"
        )

    if Part == 1:
        fig = go.Figure(
            data=[
                go.Scatter(
                    x=x,
                    y=np.real(u_numerical[time_step, :]),
                    mode="lines",
                    name="Numerical FFT Solution",
                ),
                go.Scatter(
                    x=x,
                    y=np.real(u_analytical[time_step, :]),
                    mode="lines",
                    name="Analytical Solution",
                    line=dict(dash="dash"),
                ),
            ]
        )
    elif Part == 0:
        fig = go.Figure(
            data=[
                go.Scatter(
                    x=x,
                    y=np.imag(u_numerical[time_step, :]),
                    mode="lines",
                    name="Numerical FFT Solution",
                ),
                go.Scatter(
                    x=x,
                    y=np.imag(u_analytical[time_step, :]),
                    mode="lines",
                    name="Analytical Solution",
                    line=dict(dash="dash"),
                ),
            ]
        )

    fig.update_layout(
        title="Numerisk Løsning",
        xaxis={"title": "x"},
        yaxis={"title": "Amplitude", "range": [-1, 1]},
        margin={"l": 40, "b": 40, "t": 50, "r": 10},
        legend={"x": 0, "y": 1},
        hovermode="closest",
        annotations=[
            dict(
                x=0.5,
                y=1.1,
                xref="paper",
                yref="paper",
                text=f"Time: {time_step * dt:.2f} s",
                showarrow=False,
                font=dict(size=12),
            )
        ],
    )
    return fig


@app.callback(
    Output("3d-plot", "figure"),
    [Input("compute-button", "n_clicks")],
    [
        State("C1-value", "value"),
        State("C2-value", "value"),
        State("L-value", "value"),
        State("sigma-value", "value"),
        State("v-value", "value"),
        State("theta-value", "value"),
        State("N-value", "value"),
        State("dt-value", "value"),
        State("T-value", "value"),
        State("initial-condition", "value"),
        State("Part", "value"),
    ],
)
def update_3d_plot(
    n_clicks, C1, C2, L, sigma, v, theta, N, dt, T, initial_condition, Part
):
    # Re-compute only if compute button is clicked
    ctx = callback_context
    if (
        not ctx.triggered
        or ctx.triggered[0]["prop_id"].split(".")[0] == "compute-button"
    ):
        x, u_numerical, u_analytical, steps = compute_wave_propagation(
            C1, C2, L, sigma, v, theta, N, dt, T, initial_condition, "half"
        )

    if Part == 1:
        fig_3d = {
            "data": [go.Surface(z=np.real(u_analytical))],
            "layout": go.Layout(
                title="Løsning til den ikke-lineære Schrödingerligning",
                scene=dict(
                    xaxis_title="x",
                    yaxis_title="t",
                    zaxis_title="u(x,t)",
                    xaxis=dict(
                        tickvals=[], title="x"
                    ),  # Remove x-axis tick mark values
                    yaxis=dict(
                        tickvals=[], title="t"
                    ),  # Remove y-axis tick mark values
                    zaxis=dict(
                        tickvals=[], title="u(x,t)"
                    ),  # Remove z-axis tick mark values
                ),
                width=1000,
                height=750,
                margin=dict(l=65, r=50, b=65, t=90),
            ),
        }
    elif Part == 0:
        fig_3d = {
            "data": [go.Surface(z=np.imag(u_numerical))],
            "layout": go.Layout(
                title="Løsning til den ikke-lineære Schrödingerligning",
                scene=dict(
                    xaxis_title="x",
                    yaxis_title="t",
                    zaxis_title="u(x,t)",
                    xaxis=dict(
                        tickvals=[], title="x"
                    ),  # Remove x-axis tick mark values
                    yaxis=dict(
                        tickvals=[], title="t"
                    ),  # Remove y-axis tick mark values
                    zaxis=dict(
                        tickvals=[], title="u(x,t)"
                    ),  # Remove z-axis tick mark values
                ),
                width=1000,
                height=750,
                margin=dict(l=65, r=50, b=65, t=90),
            ),
        }
    return fig_3d


@app.callback(
    [Output("interval-component", "disabled"), Output("play-pause-button", "children")],
    [Input("play-pause-button", "n_clicks")],
    [State("interval-component", "disabled")],
)
def toggle_play_pause(n_clicks, is_disabled):
    if n_clicks is None:  # If callback triggered by initial load, keep it paused
        return True, "Play"
    if n_clicks % 2 == 0:  # If clicked even number of times, keep it paused
        return True, "Play"
    else:  # If clicked odd number of times, play it
        return False, "Pause"


@app.callback(
    Output("time-step-slider", "value"),
    [Input("interval-component", "n_intervals")],
    [State("time-step-slider", "value"), State("time-step-slider", "max")],
)
def advance_time_step(n_intervals, current_value, max_value):
    new_value = (current_value + 1) % (max_value + 1)
    return new_value


# Update 'max' property of the time-step slider
@app.callback(
    Output("time-step-slider", "max"),
    [Input("compute-button", "n_clicks")],
    [State("T-value", "value"), State("dt-value", "value")],
)
def update_slider_max(n_clicks, T, dt):
    steps = int(T / dt)
    return steps - 1


@app.callback(
    Output("interval-component", "interval"), [Input("speed-slider", "value")]
)
def update_speed(value):
    # Adjusting speed value conversion to interval if needed
    return max(200 - value, 10) * 100  # Example adjustment, modify as needed


## callbacks for precision/N plot and precision/cpu time plot
@app.callback(
    [Output("precision-n-plot", "figure"), Output("precision-cpu-time-plot", "figure")],
    [Input("compute-button-2", "n_clicks")],
    [
        State("N-value-2", "value"),
        State("T-value-2", "value"),
        State("Interval-value-2", "value"),
    ],
)
def update_precision_plots(n_clicks, N, T, interval):
    C1, C2, L, sigma, v, theta, dt, initial_condition = 1, 2, 30, 0, 10, 0, 0.005, 2
    # Compute all the data arrays for the full step method first
    # compute for N=100 to N=N-value-2 in steps of Interval-value-2
    N_values = np.arange(100, N + 1, interval)
    cpu_times_full = np.zeros(N_values.size)
    cpu_times_half = np.zeros(N_values.size)
    precisions_full = np.zeros(N_values.size, dtype=np.complex64)
    precisions_half = np.zeros(N_values.size, dtype=np.complex64)
    u_numerical_full_solutions = [
        np.zeros(N_values.size, dtype=np.complex64) for _ in range(N_values.size)
    ]
    u_numerical_half_solutions = [
        np.zeros(N_values.size, dtype=np.complex64) for _ in range(N_values.size)
    ]
    u_analytical_solutions = [
        np.zeros(N_values.size, dtype=np.complex64) for _ in range(N_values.size)
    ]

    for i in range(len(N_values)):
        time_start1 = time.time()
        x, u_numerical_full, u_analytical, steps = compute_wave_propagation(
            C1, C2, L, sigma, v, theta, N_values[i], dt, T, initial_condition, "full"
        )
        time_end1 = time.time()
        cpu_times_full[i] = time_end1 - time_start1
        time_start2 = time.time()
        x, u_numerical_half, u_analytical, steps = compute_wave_propagation(
            C1, C2, L, sigma, v, theta, N_values[i], dt, T, initial_condition, "half"
        )
        time_end2 = time.time()
        cpu_times_half[i] = time_end2 - time_start2

        u_numerical_full_solutions[i] = u_numerical_full
        u_numerical_half_solutions[i] = u_numerical_half
        u_analytical_solutions[i] = u_analytical
        # compute mean squared error between the numerical and analytical solutions
        precisions_full[i] = np.abs(np.mean((u_numerical_full - u_analytical) ** 2))

        precisions_half[i] = np.abs(np.mean((u_numerical_half - u_analytical) ** 2))

        # compute the cpu time for the full step and half step methods
    # create the plots
    fig1 = go.Figure()
    fig1.add_trace(
        go.Scatter(
            x=N_values,
            y=np.real(precisions_full),
            mode="lines",
            name="Full Step Method",
        )
    )
    fig1.add_trace(
        go.Scatter(
            x=N_values,
            y=np.real(precisions_half),
            mode="lines",
            name="Half Step Method",
            line=dict(dash="dash"),
        )
    )
    fig1.update_layout(
        title="Precision vs N",
        xaxis_title="N",
        yaxis_title="Mean Squared Error",
        width=1000,
        height=750,
    )

    # plot precision vs cpu time
    fig2 = go.Figure()
    fig2.add_trace(
        go.Scatter(
            x=cpu_times_full,
            y=np.real(precisions_full),
            mode="lines",
            name="Full Step Method",
        )
    )
    fig2.add_trace(
        go.Scatter(
            x=cpu_times_half,
            y=np.real(precisions_half),
            mode="lines",
            name="Half Step Method",
            line=dict(dash="dash"),
        )
    )
    fig2.update_layout(
        title="Precision vs CPU Time",
        xaxis_title="CPU Time",
        yaxis_title="Mean squared error",
        width=1000,
        height=750,
    )

    return fig1, fig2


@app.callback(
    Output("loading-precision-n", "loading_state"),
    Output("loading-precision-cpu-time", "loading_state"),
    [Input("compute-button-2", "n_clicks")],
)
def update_loading_state(n_clicks):
    if n_clicks:
        return [
            {"is_loading": True},
            {"is_loading": True},
        ]  # Show loading state if button is clicked
    else:
        return [
            {"is_loading": False},
            {"is_loading": False},
        ]  # Hide loading state if button is not clicked


if __name__ == "__main__":
    app.run_server(port=8059, debug=True)
