import os 
os.environ['HDF5_DISABLE_VERSION_CHECK']='2'
import dash
from dash import html
from dash import dcc, Input, Output
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import dash_bootstrap_components as dbc
import sys
from graphs.ManhattanGraph import ManhattanGraph
# sys.path.insert(0,"")
# from Manhattan_Graph_Environment.gym_graphenv.envs.GraphworldManhattan import GraphEnv
# env = GraphEnv()

df_hubs = pd.read_csv("./data/hubs/longlist.csv")
ids = [i for i in range(120)]
actions = ['hub' for i in range(120)]
df_hubs['id'] = ids
df_hubs['action'] = actions

taken_steps = []
list_actions = [3,112,2]
type_of_actions = ['', 'share', 'book']

nodes =  [42437644, 42446701, 42443680, 42451674, 42437686, 42427786, 42438544, 42434800, 42429874, 7372860100, 
42453624, 449581627, 42428206, 42437881, 42434268, 42443671, 42427369, 42428183, 42428179, 42428174, 4597668026, 4597668036, 4597668023, 4597668035, 42428493, 42428491, 42428489, 42428483, 42428480, 42428476, 42428473, 42428471, 42428468, 42428464, 42428460, 42428458, 42428454, 1692433919, 42428447, 1919595915, 42428444, 42428441, 42428438, 42428436, 42428434, 42428433, 42428431, 42428428, 42428425, 42428420, 5706569905, 42428411, 42428408, 42428405, 42428402, 272195271, 2821304136, 2821304137, 2821304138, 272195270, 8262936580, 42428391, 
42428385, 42428379, 42453934, 8840333846, 8840333851, 42453952, 42430004, 42429562, 42449597, 42431611, 42445356, 42430550, 42429830, 42452817, 42448171, 4143851142, 4143851144, 42453777, 42447249, 42447246, 42440825, 
8996353563, 42452040, 42432152, 42442273, 42456197, 1919595917, 1919595925, 1919595915, 1918039880, 1918039897, 42445511, 1692433916, 42445520, 42431165, 42445534, 42439236, 42445543, 42439249, 42434140, 42443296, 42443298, 4235734225, 3892037906, 42430898, 42430903, 42454994, 42446547, 42436943, 42440743, 42440737, 42436942, 42442870, 42442862, 42430898, 42442857, 42434142, 42434201, 42442851, 42442850, 42442848, 42437333, 42442843, 42442842, 42442838, 42443127, 1918039877, 42432083, 42437113, 486869282, 42429754, 42432085, 42452067, 42445603, 42437580, 5161246301, 5161246307, 1773063789, 1773063787, 1773060097, 1773060099, 1773055865, 588455743, 4158807592, 42439813, 42437749, 42444277, 3884569931, 3884569924, 588455738, 205019740, 4145735057, 4145735066, 4147546533, 42431902, 370880739, 4145735059, 205020852, 42422270, 278608643, 42423307, 370892861, 370894980, 370897166, 370913758, 42422509, 42457925, 42457940, 42445392, 42445390, 42445387, 5131026388, 42445382, 
42445378, 42445374, 3786901743, 561042190, 42445365]

hub_node_ids = [42423296, 42423307, 42437644, 42427915, 371188750, 371188756, 371239958, 42449945, 42433058, 42450468, 42440743, 1825841704, 42433066, 42445867, 42445357, 5426969134, 2054405680, 370912817, 42428980, 9140654137, 370892861, 42445374, 42427965, 42427968, 42427970, 42427972, 42445392, 42445916, 42440798, 561035358, 371188320, 42435684, 42450025, 42446959, 371207282, 406006393, 277482105, 42446977, 370894980, 42449029, 370924677, 4779073677, 4779073680, 42438809, 4320028826, 7490266268, 42440350, 42444457, 42442415, 205020852, 42442937, 205024444, 42450634, 3843180751, 371209940, 589099734, 42424025, 42445018, 3579432156, 42442463, 42442475, 370914027, 42437358, 370915061, 42423549, 42423039, 42451712, 42447105, 278608643, 42423051, 595295501, 42428174, 42443534, 370897166, 561042199, 42447132, 42430253, 42457401, 595352904, 42439497, 42424145, 42432856, 42430304, 5779545445, 4886250352, 42454391, 42427769, 42432889, 42455929, 42430333, 42450820, 42436486, 42433422, 1919595922, 4145735059, 42430361, 42443674, 370924957, 42429342, 370888100, 42430375, 42453934, 42428863, 595314119, 589929417, 1061531596, 42427863, 42432985, 42423774, 370913758, 3099327970, 9177424868, 248708582, 3099327976, 100522479, 42426865, 370913779, 596775930, 370898427, 1692433919]

manhattan_graph = ManhattanGraph(filename='simple', num_hubs=120)

df_route = pd.DataFrame()
df_route['longitude'] = [0 for i in range(len(nodes))]
df_route['latitude'] = [0 for i in range(len(nodes))]

for i in range(len(nodes)):
    results = manhattan_graph.get_coordinates_of_node(nodes[i])
    df_route['longitude'][i] = results[0]
    df_route['latitude'][i] = results[1]


# Function to create map
def create_map_from_df(df_hubs, df_route, hubs=[]):

    px.set_mapbox_access_token(open("Manhattan_Graph_Environment/mapbox_token").read())
    fig = px.scatter_mapbox(df_hubs, lat="latitude", lon="longitude", hover_name ="id", color="action", #size="car_hours",
                    color_continuous_scale=px.colors.cyclical.IceFire, size_max=30, zoom=11)

    if(len(hubs)!=0):
        fig.add_trace(go.Scattermapbox(
            mode = "lines",
            lon = [df_route['longitude'][i] for i in range(len(df_route['longitude']))],
            lat = [df_route['latitude'][i] for i in range(len(df_route['latitude']))],
            marker = {'size': 10},
            hovertext  = [hub_node_ids.index(n) if n in hub_node_ids else ' ' for n in nodes]))
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

    df_hubs['action'] = actions # start hub

    shared_hubs = {'1':7, '2':10, '3':-5}
    return dcc.Graph(figure=create_map_from_df(df_hubs, df_route, taken_steps), id='my-graph'), [html.Div("{}:  {} km".format( i, shared_hubs[i])) for i in shared_hubs]


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
       dcc.Graph( figure=create_map_from_df(df_hubs, df_route), id='my-graph')

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
       dcc.Graph( figure=create_map_from_df(df_hubs, df_route), id='my-graph')

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

    df_hubs['action'][start_hub]='start'
    df_hubs['action'][final_hub]='final'
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
    return html.Div('CURRENT ORDER: {} -> {}'.format(start_hub,final_hub)), dcc.Graph( figure=create_map_from_df(df_hubs, df_route, list_actions),id='my-graph'), 'Wait: {}'.format(nr_wait), 'Book: {}'.format(nr_book), 'Share: {}'.format(nr_shared), route_string


@app.callback(
    Output('destination-hub-2', 'children'),
    #Output(component_id='map-2', component_property='children'),
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

    df_hubs['action'][start_hub]='start'
    df_hubs['action'][final_hub]='final'
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