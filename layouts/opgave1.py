import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objs as go
from dash import Dash, Input, Output, State, callback_context, dcc, html

layout = html.Div(
    [
        dcc.Markdown(children=markdown_text, mathjax=True, style={"fontSize": 24}),
        html.Div(
            [
                html.Div(
                    [
                        html.Label("L:"),
                        dcc.Input(
                            id="L-value",
                            type="number",
                            value=initial_L,
                            step=1,
                            style={"width": "105%"},
                        ),
                    ],
                    className="one column",
                    style={"margin-right": "10px"},
                ),
                html.Div(
                    [
                        html.Label("w:"),
                        dcc.Input(
                            id="w-value",
                            type="number",
                            value=initial_w,
                            step=1,
                            style={"width": "105%"},
                        ),
                    ],
                    className="one column",
                    style={"margin-right": "10px"},
                ),
                html.Div(
                    [
                        html.Label("k0:"),
                        dcc.Input(
                            id="k0-value",
                            type="number",
                            value=initial_k0,
                            step=1,
                            style={"width": "105%"},
                        ),
                    ],
                    className="one column",
                    style={"margin-right": "10px"},
                ),
                html.Div(
                    [
                        html.Label("v:"),
                        dcc.Input(
                            id="v-value",
                            type="number",
                            value=initial_v,
                            step=0.1,
                            style={"width": "105%"},
                        ),
                    ],
                    className="one column",
                    style={"margin-right": "10px"},
                ),
                html.Div(
                    [
                        html.Label("N:"),
                        dcc.Input(
                            id="N-value",
                            type="number",
                            value=initial_N,
                            step=1,
                            style={"width": "105%"},
                        ),
                    ],
                    className="one column",
                    style={"margin-right": "10px"},
                ),
                html.Div(
                    [
                        html.Label("dt:"),
                        dcc.Input(
                            id="dt-value",
                            type="number",
                            value=initial_dt,
                            step=0.001,
                            style={"width": "105%"},
                        ),
                    ],
                    className="one column",
                    style={"margin-right": "10px"},
                ),
                html.Div(
                    [
                        html.Label("T:"),
                        dcc.Input(
                            id="T-value",
                            type="number",
                            value=initial_T,
                            step=1,
                            style={"width": "105%"},
                        ),
                    ],
                    className="one column",
                    style={"margin-right": "10px"},
                ),
                html.Div(
                    [
                        html.Button(
                            "Compute",
                            id="compute-button",
                            n_clicks=0,
                            style={"margin-top": "10px"},
                        )
                    ],
                    className="one column",
                    style={"margin-top": "22px"},
                ),
            ],
            className="row",
            style={"margin-bottom": "20px"},
        ),
        dcc.Graph(id="wave-animation"),  # This needs to be in the layout
        html.Div(
            [
                html.Button(
                    "Play/Pause",
                    id="play-pause-button",
                    n_clicks=0,
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
                    min=200,
                    max=1000,
                    value=250,
                    step=10,
                    marks={i: f"{i}ms" for i in range(200, 1000, 100)},
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
    ]
)


@app.callback(
    Output("wave-animation", "figure"),
    [Input("time-step-slider", "value"), Input("compute-button", "n_clicks")],
    [
        State("L-value", "value"),
        State("w-value", "value"),
        State("k0-value", "value"),
        State("v-value", "value"),
        State("N-value", "value"),
        State("dt-value", "value"),
        State("T-value", "value"),
    ],
)
def update_output(time_step, n_clicks, L, w, k0, v, N, dt, T):

    # Re-compute only if compute button is clicked
    ctx = callback_context
    if (
        not ctx.triggered
        or ctx.triggered[0]["prop_id"].split(".")[0] == "compute-button"
    ):
        global x, u_numerical, u_moving, steps
        x, u_numerical, u_moving, steps = compute_wave_propagation(
            L, w, k0, v, N, dt, T
        )

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
                y=u_moving[time_step, :],
                mode="lines",
                name="Analytical Solution",
                line=dict(dash="dash"),
            ),
        ]
    )
    fig.update_layout(
        title="Reeldel plottet",
        xaxis={"title": "x"},
        yaxis={"title": "Amplitude", "range": [-1, 1]},
        margin={"l": 40, "b": 40, "t": 50, "r": 10},
        legend={"x": 0, "y": 1},
        hovermode="closest",
    )
    return fig


@app.callback(
    [Output("interval-component", "disabled"), Output("play-pause-button", "children")],
    [Input("play-pause-button", "n_clicks")],
    [State("interval-component", "disabled")],
)
def toggle_play_pause(n_clicks, is_disabled):
    if is_disabled:
        return False, "Pause"
    else:
        return True, "Play"


@app.callback(
    Output("time-step-slider", "value"),
    [Input("interval-component", "n_intervals")],
    [State("time-step-slider", "value"), State("time-step-slider", "max")],
)
def advance_time_step(n_intervals, current_value, max_value):
    new_value = (current_value + 1) % (max_value + 1)
    return new_value


@app.callback(
    Output("interval-component", "interval"), [Input("speed-slider", "value")]
)
def update_speed(value):
    # Adjusting speed value conversion to interval if needed
    return max(200 - value, 10) * 10  # Example adjustment, modify as needed


# Update 'max' property of the time-step slider
@app.callback(
    Output("time-step-slider", "max"),
    [Input("compute-button", "n_clicks")],
    [State("T-value", "value"), State("dt-value", "value")],
)
def update_slider_max(n_clicks, T, dt):
    steps = int(T / dt)
    return steps - 1
