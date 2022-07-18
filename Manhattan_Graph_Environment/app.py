import dash
from dash import html
from dash import dcc, Input, Output
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import dash_bootstrap_components as dbc

# load and manipulate data
df = pd.read_csv("./data/hubs/longlist.csv")
ids = [i for i in range(120)]
actions = ['book' for i in range(120)]
actions[0] = 'share' # share
actions[1] = 'share' # share
actions[2] = 'share' # share
actions[3] = 'share' # share
actions[4] = 'current' # current pos
actions[5] = 'final' # final hub
actions[6] = 'start' # start hub
df['id'] = ids
df['action'] = actions

# Function to create map
def create_map_from_df(df):
    px.set_mapbox_access_token(open("Manhattan_Graph_Environment/mapbox_token").read())
    fig = px.scatter_mapbox(df, lat="latitude", lon="longitude", hover_name ="id", color="action", #size="car_hours",
                    color_continuous_scale=px.colors.cyclical.IceFire, size_max=30, zoom=10)

    fig.add_trace(go.Scattermapbox(
        mode = "lines",
        lon = [df['longitude'][0], df['longitude'][1], df['longitude'][2], df['longitude'][3], df['longitude'][4]],
        lat = [df['latitude'][0], df['latitude'][1], df['latitude'][2], df['latitude'][3], df['latitude'][4]],
        marker = {'size': 10}))
    #fig.show()

    return fig


app = dash.Dash(__name__)


app = dash.Dash(__name__)

app.layout = html.Div(children=[

html.H1(children='Hitchhike Dashboard'),
html.Div(children=[
       dcc.Graph( figure=create_map_from_df(df))

    ], className='left-dashboard', id='map'),

    html.Div(children=[

        html.P('Current Order: ', id='destination-hub'),
        html.H4('Available Shared Rides:'),
        html.P('---Destination Reduction of distance to find hub---'),
        html.Div(id='shared'),
        html.Div(id='hidden-div', style={'display':'none'}),
        html.H4('Actions taken:'),
        html.Div(id='actions-taken'),
        html.Div([
        "Next step HUB: ",
        dcc.Input(id='next-hub-input', value= None, type='number')
    ], className='right-input'),

    ], className='right-dashboard'),
   
    html.Div([
        "Destination HUB: ",
        dcc.Input(id='dest-hub-input', value=None, type='text')
    ])

], style={'width': '100%', 'display': 'inline-block'}
)

# @app.callback(
#     Output(component_id='destination-hub', component_property='children'),
#     Output(component_id='shared', component_property='children'),
#     Output(component_id='actions-taken', component_property='children'),
#     Input(component_id='dest-hub-input', component_property='value'),
#     prevent_initial_call=True
# )
# def make_order(input_value):
#     shared_hubs = {'1':7, '2':10, '3':-5}
#     wait =1
#     share = 0
#     book = 0
#     return f'TO: {input_value}', [html.Div("{}:  {} km".format( i, shared_hubs[i])) for i in shared_hubs], html.Div([html.Div('Wait: {}'.format(wait)), html.Div('Book: {}'.format(book)), html.Div('Share: {}'.format(share))])

# @app.callback(
#     Output(component_id='hidden-div', component_property='children'),
#     Input(component_id='dest-hub-input', component_property='value'),
#     prevent_initial_call=True
# )
# def make_order(input):
#     next_step(input)
#     return


@app.callback(
    Output(component_id='map', component_property='children'),
    Output(component_id='shared', component_property='children'),
    Output(component_id='actions-taken', component_property='children'),
    Input(component_id='next-hub-input', component_property='value'),
    prevent_initial_call=True
)
def next_step(input_value):

    actions = ['book' for i in range(120)]
    actions[0] = 'share' # share
    actions[1] = 'share' # share
    actions[2] = 'share' # share
    actions[3] = 'share' # share
    actions[4] = 'current' # current pos
    actions[5] = 'final' # final hub
    actions[int(input_value)] = 'start' # start hub

    df['action'] = actions # start hub

    shared_hubs = {'1':7, '2':10, '3':-5}
    wait =1
    share = 0
    book = 0
    return html.Div(children=[
       dcc.Graph( figure=create_map_from_df(df))

    ], className='left-dashboard'), [html.Div("{}:  {} km".format( i, shared_hubs[i])) for i in shared_hubs], html.Div([html.Div('Wait: {}'.format(wait)), html.Div('Book: {}'.format(book)), html.Div('Share: {}'.format(share))])

 
if __name__ == "__main__":
    app.run_server(debug=True)
