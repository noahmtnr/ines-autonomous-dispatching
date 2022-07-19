import os 
os.environ['HDF5_DISABLE_VERSION_CHECK']='2'
import dash
from dash import html
from dash import dcc, Input, Output
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import dash_bootstrap_components as dbc

df = pd.read_csv("./data/hubs/longlist.csv")
ids = [i for i in range(120)]
actions = ['hub' for i in range(120)]
df['id'] = ids
df['action'] = actions

taken_steps = []
list_actions = [3,112,2]
type_of_actions = ['', 'share', 'book']


# Function to create map
def create_map_from_df(df, hubs=[]):
    px.set_mapbox_access_token(open("Manhattan_Graph_Environment/mapbox_token").read())
    fig = px.scatter_mapbox(df, lat="latitude", lon="longitude", hover_name ="id", color="action", #size="car_hours",
                    color_continuous_scale=px.colors.cyclical.IceFire, size_max=30, zoom=11)

    if(len(hubs)!=0):
        fig.add_trace(go.Scattermapbox(
            mode = "lines",
            lon = [df['longitude'][i] for i in hubs],
            lat = [df['latitude'][i] for i in hubs],
            marker = {'size': 10},
            hovertext  = [df['id'][i] for i in hubs]))
    return fig

app = dash.Dash(__name__,  suppress_callback_exceptions = True)

@app.callback(
    Output(component_id='map-2', component_property='children'),
    Output(component_id='shared', component_property='children'),
    Input(component_id='next-hub-input', component_property='value'),
    prevent_initial_call=True
)
def next_step(input_value):

    if input_value is None: 
        return dash.no_update

    taken_steps.append(input_value)

    actions = ['book' for i in range(120)]
    actions[0] = 'share' # share
    actions[1] = 'share' # share
    actions[2] = 'share' # share
    actions[3] = 'share' # share
    actions[input_value] = 'current' # current pos
    actions[5] = 'final' # final hub
    actions[6] = 'start' # start hub

    df['action'] = actions # start hub

    shared_hubs = {'1':7, '2':10, '3':-5}
    return dcc.Graph(figure=create_map_from_df(df, taken_steps), id='my-graph'), [html.Div("{}:  {} km".format( i, shared_hubs[i])) for i in shared_hubs]


app.layout = html.Div([
    html.H1('Hitchhike Dashboard'),
    dcc.Tabs(id="tabs-example", value='tab-1-example', children=[
        dcc.Tab(label='Static Visualization', value='tab-1-example', className = 'tab-label'),
        dcc.Tab(label='Interactive Visualization', value='tab-2-example', className = 'tab-label'),
    ], colors = { "border": '#333399', "primary": 'blueviolet', "background": '#ccb3ff'}),
    html.Div(id='tabs-content-example')
])

@app.callback(Output('tabs-content-example', 'children'),
              Input('tabs-example', 'value'))
def render_content(tab):
    if tab == 'tab-1-example':
        return html.Div(children=[
html.Div([html.Button('Start', id='start-button-1', n_clicks=0)]),
html.Div(children=[
       dcc.Graph( figure=create_map_from_df(df), id='my-graph')

    ], className='left-dashboard', id='map-1'),

    html.Div(children=[

        html.H4('CURRENT ORDER: ', id='destination-hub-1'),
        html.H4('Calculated route: ',  id='calc-route-1'),
        html.H4('Actions taken:', id= 'actions-taken-titel'),
        html.Div(className = 'grid-container', id='wait-1'),
        html.Div(className = 'grid-container', id='share-1'),
        html.Div(className = 'grid-container', id='book-1'),
    ], className='right-dashboard'),
     
], className = 'main-body'
)

    elif tab == 'tab-2-example':
        return html.Div(children=[
html.Div([html.Button('Start', id='start-button-2', n_clicks=0)]),
html.Div(children=[
       dcc.Graph( figure=create_map_from_df(df), id='my-graph')

    ], className='left-dashboard', id='map-2'),

    html.Div(children=[

        html.H4('CURRENT ORDER: ', id='destination-hub-2'),
        html.H4('Calculated route: ',  id='calc-route-2'),
               
        html.H4('Actions taken:', id= 'actions-taken-titel'),
        html.Div(className = 'grid-container', id='wait-2'),
        html.Div(className = 'grid-container', id='share-2'),
        html.Div(className = 'grid-container', id='book-2'),
        
        html.Div( children=[
            html.Div(html.H3('Step by Step Analysis: '), id = 'titel-analysis'),
            html.Div([
        "Next step HUB: ", dcc.Input(id='next-hub-input', value= None, type='number', debounce=True),
    ], className='right-input')], id = 'step-by-step'), 
    html.H4('Available Shared Rides:', id = 'available-shared'),  
    html.Div(id='shared'),   

    ], className='right-dashboard')

], className = 'main-body'
)

@app.callback(
    Output('destination-hub-1', 'children'),
    Output(component_id='map-1', component_property='children'),
    Output(component_id='wait-1', component_property='children'),
    Output(component_id='share-1', component_property='children'),
    Output(component_id='book-1', component_property='children'),
    Output(component_id='calc-route-1', component_property='children'),
    Input('start-button-1', 'n_clicks'),
    prevent_initial_call=True
)
def start_order_1(n_clicks):
    start_hub = list_actions[0]
    final_hub = list_actions[-1]

    df['action'][start_hub]='start'
    df['action'][final_hub]='final'
    nr_wait = 0
    nr_shared = 0
    nr_book = 0
    route_string='Calculated route: '

    for i in type_of_actions:
        if i == 'wait':
            nr_wait+=1
        if i == 'share':
            nr_shared+=1
        if i == 'book':
            nr_book+=1
    for i in range(len(list_actions)):
        route_string+= f'{list_actions[i]} -> '
    route_string = route_string[0:-3]

    return html.Div('CURRENT ORDER: {} -> {}'.format(start_hub,final_hub)), dcc.Graph( figure=create_map_from_df(df, list_actions),id='my-graph'), 'Wait: {}'.format(nr_wait), 'Book: {}'.format(nr_book), 'Share: {}'.format(nr_shared), route_string


@app.callback(
    Output('destination-hub-2', 'children'),
    #Output(component_id='map', component_property='children'),
    Output(component_id='wait-2', component_property='children'),
    Output(component_id='share-2', component_property='children'),
    Output(component_id='book-2', component_property='children'),
    Output(component_id='calc-route-2', component_property='children'),
    Input('start-button-2', 'n_clicks'),
    prevent_initial_call=True
)
def start_order_2(n_clicks):
    start_hub = list_actions[0]
    final_hub = list_actions[-1]
    global taken_steps
    taken_steps = []

    df['action'][start_hub]='start'
    df['action'][final_hub]='final'
    nr_wait = 0
    nr_shared = 0
    nr_book = 0
    route_string='Calculated route: '

    for i in type_of_actions:
        if i == 'wait':
            nr_wait+=1
        if i == 'share':
            nr_shared+=1
        if i == 'book':
            nr_book+=1

    for i in range(len(list_actions)):
        route_string+= f'{list_actions[i]} -> '
    route_string = route_string[0:-3]
    return html.Div('CURRENT ORDER: {} -> {}'.format(start_hub,final_hub)), 'Wait: {}'.format(nr_wait), 'Book: {}'.format(nr_book), 'Share: {}'.format(nr_shared), route_string



if __name__ == '__main__':
    app.run_server(debug=True)