import dash
from dash import html

dash.register_page(__name__, path="/")

layout = html.Div(
    [
        html.H1("Welcome to the Home Page"),
        html.Div("This is the home page of the Dash app."),
    ]
)
