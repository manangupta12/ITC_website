import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.graph_objs as go
from dash import Dash
from dash.dependencies import Input, State, Output
from .Dash_fun import apply_layout_with_auth, load_object, save_object

url_base = '/dash/app4/'
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

layout = html.Div(children=[
    html.H1(children='Inferences or Suggestions related to product'),

    html.Div(children='')
])

def Add_Dash(server):
    app = Dash(server=server, url_base_pathname=url_base)
    apply_layout_with_auth(app, layout)
    return app.server