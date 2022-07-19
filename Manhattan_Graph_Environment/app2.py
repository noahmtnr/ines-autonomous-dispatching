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
actions = ['hub' for i in range(120)]
df['id'] = ids
df['action'] = actions

taken_steps = []

# define list containing hubs/actions that were taken (will later come from tested agent)
# list_actions = [3,20,20,5,7,7,7,110,112,112,2]
# type_of_actions = ['','book','wait','share', 'share', 'wait', 'wait', 'share', 'share', 'wait', 'book']

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


app = dash.Dash(__name__)

@app.callback(
    Output(component_id='map', component_property='children'),
    Output(component_id='shared', component_property='children'),
    Input(component_id='next-hub-input', component_property='value'),
    prevent_initial_call=True
)
def next_step(input_value):

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
    wait =1
    share = 0
    book = 0
    return dcc.Graph(figure=create_map_from_df(df, taken_steps), id='my-graph'), [html.Div("{}:  {} km".format( i, shared_hubs[i])) for i in shared_hubs]



app.layout = html.Div(children=[

html.H1(children='Hitchhike Dashboard'),
html.Div([html.Button('Start', id='start-button', n_clicks=0)]),
html.Div(children=[
       dcc.Graph( figure=create_map_from_df(df), id='my-graph')

    ], className='left-dashboard', id='map'),

    html.Div(children=[

        html.H4('CURRENT ORDER: ', id='destination-hub'),
        html.H4('Calculated route: ',  id='calc-route'),
        
        # html.P('---Destination Reduction of distance to find hub---'),
       
        html.H4('Actions taken:', id= 'actions-taken-titel'),
        html.Div(className = 'grid-container', id='wait'),
        html.Div(className = 'grid-container', id='share'),
        html.Div(className = 'grid-container', id='book'),
       
        
        html.Div( children=[
            html.Div(html.H3('Step by Step Analysis: '), id = 'titel-analysis'),
            html.Div([
        "Next step HUB: ", dcc.Input(id='next-hub-input', value= None, type='number', debounce=True),
    ], className='right-input')], id = 'step-by-step'), 
    html.H4('Available Shared Rides:', id = 'available-shared'),  
    html.Div(id='shared'),   

    ], className='right-dashboard')
   
    # html.Div([
    #     "Destination HUB: ",
    #     dcc.Input(id='dest-hub-input', value=None, type='text')
    # ])

], className = 'main-body'
)



@app.callback(
    Output('destination-hub', 'children'),
    #Output(component_id='map', component_property='children'),
    Output(component_id='wait', component_property='children'),
    Output(component_id='share', component_property='children'),
    Output(component_id='book', component_property='children'),
    Output(component_id='calc-route', component_property='children'),
    Input('start-button', 'n_clicks'),
    prevent_initial_call=True
)
def start_order(n_clicks):
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

if __name__ == "__main__":
    app.run_server(debug=True)
