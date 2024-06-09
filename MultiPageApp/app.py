import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, dcc, html

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Multi-Tab Dash App"

# Import layouts and callback registration functions from the individual files
from apps.opgave1 import layout as layout1
from apps.opgave1 import register_callbacks as register_callbacks1
from apps.opgave2 import layout as layout2
from apps.opgave2 import register_callbacks as register_callbacks2
from apps.opgave3 import layout as layout3
from apps.opgave3 import register_callbacks as register_callbacks3
from apps.opgave4 import layout as layout4
from apps.opgave4 import register_callbacks as register_callbacks4
from apps.projektplan_opgave1 import layout as layout5
from apps.projektplan_opgave1 import register_callbacks as register_callbacks5
from apps.projektplan_opgave2 import layout as layout6
from apps.projektplan_opgave2 import register_callbacks as register_callbacks6

# Define the layout for the main app
app.layout = html.Div(
    [
        dcc.Tabs(
            id="tabs",
            children=[
                dcc.Tab(label="Opgave 1", value="opgave1"),
                dcc.Tab(label="Opgave 2", value="opgave2"),
                dcc.Tab(label="Opgave 3", value="opgave3"),
                dcc.Tab(label="Opgave 4", value="opgave4"),
                dcc.Tab(label="Projektplan Opgave 1", value="projektplan_opgave1"),
                dcc.Tab(label="Projektplan Opgave 2", value="projektplan_opgave2"),
            ],
        ),
        html.Div(id="content"),
    ]
)


# Update content based on selected tab
@app.callback(Output("content", "children"), [Input("tabs", "value")])
def render_content(tab):
    if tab == "opgave1":
        return layout1
    elif tab == "opgave2":
        return layout2
    elif tab == "opgave3":
        return layout3
    elif tab == "opgave4":
        return layout4
    elif tab == "projektplan_opgave1":
        return layout5
    elif tab == "projektplan_opgave2":
        return layout6


# Register callbacks for each app
register_callbacks1(app)
register_callbacks2(app)
register_callbacks3(app)
register_callbacks4(app)
register_callbacks5(app)
register_callbacks6(app)

if __name__ == "__main__":
    app.run_server(debug=True)
