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
import ast
from graphs.ManhattanGraph import ManhattanGraph
import pickle
# sys.path.insert(0,"")
from gym_graphenv.envs.GraphworldManhattan import GraphEnv
env = GraphEnv(use_config=True)
env.reset()

# read test dataframe 
filepath = "test_orders_dashboard.csv"
df_test = pd.read_csv(filepath)
#df_test = pd.read_csv('D:/ines-autonomous-dispatching/Manhattan_Graph_Environment/test_orders_dashboard.csv')
for i in range(len(df_test['Hubs'])):
    df_test['Hubs'][i] = ast.literal_eval(df_test['Hubs'][i])
    df_test['Actions'][i] = ast.literal_eval(df_test['Actions'][i])
    df_test['Nodes'][i] = ast.literal_eval(df_test['Nodes'][i])

df_hubs = pd.read_csv("./data/hubs/longlist.csv")
ids = [i for i in range(120)]
actions = ['hub' for i in range(120)]
df_hubs['id'] = ids
df_hubs['action'] = actions

hub_node_ids = [42423039, 42423051, 42423296, 42423307, 42423549, 42423774, 42424025, 42424145, 42426865, 42427769, 42427863, 42427915, 42427965, 42427968, 42427970, 42427972, 42428174, 42428863, 42428980, 42429342, 42430253, 42430304, 42430333, 42430361, 42430375, 42432856, 42432889, 42432985, 42433058, 42433066, 42433422, 42435684, 42436486, 42437358, 42437644, 42438809, 42439497, 42440350, 42440743, 42440798, 42442415, 42442463, 42442475, 42442937, 42443534, 42443674, 42444457, 42445018, 424457, 370913758, 370913779, 370914027, 370915061, 370924677, 370924957, 371188320, 371188750, 371188756, 371207282, 371209940, 371239958, 406006393, 561035358, 561042199, 589099734, 589929417, 595295501, 595314119, 595352904, 596775930, 1061531596, 1692433919, 1825841704, 1919595922, 2054405680, 3099327970, 3099327976, 3579432156, 3843180751, 4145735059, 4320028826, 4779073677, 4779073680, 4886250352, 5426969134, 5779545445, 7490266268, 9140654137, 9177424868]

manhattan_graph = ManhattanGraph(filename='simple', num_hubs=120)

# Function to create map
def create_map_from_df(df_hubs, df_route=pd.DataFrame(), test_id=0):

    px.set_mapbox_access_token(open("Manhattan_Graph_Environment/mapbox_token").read())
    fig = px.scatter_mapbox(df_hubs, lat="latitude", lon="longitude", hover_name ="id", color="action", #size="car_hours",
                    color_continuous_scale=px.colors.cyclical.IceFire, size_max=30, zoom=11)

    if(df_route.empty == False):
        fig.add_trace(go.Scattermapbox(
            mode = "lines",
            lon = [df_route['longitude'][i] for i in range(len(df_route['longitude']))],
            lat = [df_route['latitude'][i] for i in range(len(df_route['latitude']))],
            marker = {'size': 10},
            hovertext  = [manhattan_graph.get_hub_index_by_nodeid(n) for n in df_route['node_id']]))
    return fig


# TODO: change initial values to 0
def create_piechart(wait=1, share=1, book=1):
    colors = ['LightSteelBlue', 'Gainsboro', 'LightSlateGrey']
    labels = ['wait','share','book']
    values = [wait,share,book]
    # pull is given as a fraction of the pie radius
    fig = go.Figure(data=[go.Pie(labels=labels, values=values, pull=[0, 0.1, 0], text=labels, textposition='inside', textfont_size=18, showlegend=False, hoverinfo='text+value+percent', textinfo='value+text')])
    fig.update_traces(marker=dict(colors=colors))
    fig.update_layout(margin=dict(t=0, b=0, l=0, r=0))

    return fig

app = dash.Dash(__name__,  suppress_callback_exceptions = True)


#to be modified (calculate nodes between hubs with step() function)
@app.callback(
    Output(component_id='map-2', component_property='children'),
    #Output(component_id='shared', component_property='children'),
    # TODO: show line below
    #Output(component_id='div-piechart2', component_property='children'),
    Input(component_id='next-hub-input', component_property='value'),
    prevent_initial_call=True
)
def next_step(input_value):

    if input_value is None: 
        return dash.no_update


    shared_rides = list()
    shared_ids = list()
    state, reward,  done, info = env.step(input_value)

    rem_distance = state['remaining_distance']


    print('Trips: ', env.available_actions)
    print('Position: ', env.position)
    taken_steps.extend(info['route'])

    df_route = pd.DataFrame()
    df_route['longitude'] = [0.0 for i in range(len(taken_steps))]
    df_route['latitude'] = [0.0 for i in range(len(taken_steps))]
    df_route['node_id'] = [0 for i in range(len(taken_steps))]

    for i in range(len(taken_steps)):
        results = manhattan_graph.get_coordinates_of_node(taken_steps[i])
        #print(results)
        df_route['longitude'][i] = results[0]
        df_route['latitude'][i] = results[1]
        df_route['node_id'][i] = taken_steps[i]
    #print(df_route)

    #convert node ids list in df_route with coordinates
    trips = env.availableTrips()
    print('Trips: ',trips)
        
    for i, trip in enumerate(trips):
        shared_ids.append(trip['target_hub'])
    print('Shared: ', shared_ids)
        
    all_hubs = env.hubs

    book_own_ids = list(set(all_hubs) - set(shared_ids))
    
    position = env.manhattan_graph.get_nodeid_by_hub_index(env.position)
    final = env.manhattan_graph.get_nodeid_by_hub_index(env.final_hub)
    start = env.manhattan_graph.get_nodeid_by_hub_index(env.start_hub)

    actions = []
    #print(all_hubs)
    for n in all_hubs:
        if n == position:
            actions.append('position')
        else:
            if n == final:
                actions.append('final')
            else:
                if n == start:
                    actions.append('start')
                else:
                    if n in shared_ids:
                        print('Aici e share', n)
                        actions.append('shared')
                    else:
                        if n in book_own_ids:
                            actions.append('book') 
    df_hubs['action'] = actions

    ###
    # either take current action or look it up in df_test and then count up actions
    # current_action = df_hubs['action'][input_value]
    # previous_test_case = -1
    # current_test_case = test_id
    # # initialize all actions with 0 if test case changes
    # if previous_test_case != current_test_case:
    #     number_wait = 0
    #     number_book = 0
    #     number_share = 0
    # else:
    #     if current_action == 'position':
    #         number_wait += 1
    #     elif current_action == 'book':
    #         number_book += 1
    #     else:
    #         number_share += 1

    #to modify
    return dcc.Graph(figure=create_map_from_df(df_hubs, df_route, test_id), id='my-graph')#, dcc.Graph(figure=create_piechart(number_wait,number_shared,number_book), id='graph_actions2')


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
#html.Div([html.Button('Start', id='start-button-1', n_clicks=0)]),
html.Div(children=[
       dcc.Graph( figure=create_map_from_df(df_hubs), id='my-graph')

    ], className='left-dashboard', id='map-1'),

    html.Div(children=[
        html.Div(dcc.Dropdown(['Test 1', 'Test 2', 'Test 3'], placeholder="Select an order", id='dropdown1')),

        html.H4('CURRENT ORDER: ', id='destination-hub-1'),
        html.H4('Calculated route: ',  id='calc-route-1'),
        html.H4('Actions taken:', id= 'actions-taken-titel'),
        html.Div(dcc.Graph(figure=create_piechart(), id='graph_actions'), id='div-piechart'),
        # html.Div(className = 'grid-container', id='wait-1'),
        # html.Div(className = 'grid-container', id='share-1'),
        # html.Div(className = 'grid-container', id='book-1'),
    ], className='right-dashboard'),
     
], className = 'main-body'
)

    elif tab == 'tab-2-example':
        return html.Div(children=[
#html.Div([html.Button('Start', id='start-button-2', n_clicks=0)]),
html.Div(children=[
       dcc.Graph( figure=create_map_from_df(df_hubs), id='my-graph')

    ], className='left-dashboard', id='map-2'),

    html.Div(children=[
        html.Div(dcc.Dropdown(['Test 1', 'Test 2', 'Test 3'], placeholder="Select an order", id='dropdown2')),

        html.H4('CURRENT ORDER: ', id='destination-hub-2'),
        html.H4('Calculated route: ',  id='calc-route-2'),
               
        html.H4('Actions taken:', id= 'actions-taken-titel'),
        #html.Div(dcc.Graph(figure=create_piechart(), id='graph_actions2'), id='div-piechart2'),
        # TODO: delete following 3 lines and show upper line
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
    #Output()
    Output('destination-hub-1', 'children'),
    Output(component_id='map-1', component_property='children'),
    # Output(component_id='wait-1', component_property='children'),
    # Output(component_id='share-1', component_property='children'),
    # Output(component_id='book-1', component_property='children'),
    Output(component_id='calc-route-1', component_property='children'),
    #Output(component_id='actions-taken-titel', component_property='children'),
    Output(component_id='div-piechart', component_property='children'),
    Input('dropdown1', 'value'),
    ## replace start button with drop down 
    #Input('start-button-1', 'n_clicks'),
    prevent_initial_call=True
)
def start_order_1(value):
    # hier jetzt in dataframe schauen bei id
    if(value =='Test 1'):
        test_id = 0
    else:
        if(value == 'Test 2'):
            test_id = 1
        else:
            if(value == 'Test 3'):
                test_id = 2

    start_hub = df_test['Hubs'][test_id][0] #list_actions[0] 
    final_hub = df_test['Hubs'][test_id][-1] #list_actions[-1]

    # manipulate hubs dataframe based on start and final hub of test case
    df_hubs['action'][start_hub] = 'start'
    df_hubs['action'][final_hub] = 'final'

    nr_wait = 0
    nr_shared = 0
    nr_book = 0
    route_string='Calculated route: '

    for i in df_test['Actions'][test_id]:
        if i == 'Wait':
            nr_wait+=1
        if i == 'Share':
            nr_shared+=1
        if i == 'Book':
            nr_book+=1
    print("Wait:",nr_wait)
    print("Share:",nr_shared)
    print("Book:",nr_book)
    for i in range(len(df_test['Hubs'][test_id])):
        route_string += str(df_test['Hubs'][test_id][i]) + ' ->'
    route_string = route_string[0:-3]

    df_route = pd.DataFrame()
    df_route['longitude'] = [0.0 for i in range(len(df_test['Nodes'][test_id]))]
    df_route['latitude'] = [0.0 for i in range(len(df_test['Nodes'][test_id]))]
    df_route['node_id'] = [0 for i in range (len(df_test['Nodes'][test_id]))]

    for i in range(len(df_test['Nodes'][test_id])):
        results = manhattan_graph.get_coordinates_of_node(df_test['Nodes'][test_id][i])
        #print(results)
        df_route['longitude'][i] = results[0]
        df_route['latitude'][i] = results[1]
        df_route['node_id'][i] = df_test['Nodes'][test_id][i]
    #print(df_route)

    return html.Div('CURRENT ORDER: {} -> {}'.format(start_hub, final_hub)), dcc.Graph( figure=create_map_from_df(df_hubs, df_route, test_id),id='my-graph'), route_string, dcc.Graph(figure=create_piechart(nr_wait,nr_shared,nr_book), id='graph_actions')


@app.callback(
    Output('destination-hub-2', 'children'),
    #Output(component_id='map-2', component_property='children'),
    # Output(component_id='wait-2', component_property='children'),
    # Output(component_id='share-2', component_property='children'),
    # Output(component_id='book-2', component_property='children'),
    Output(component_id='calc-route-2', component_property='children'),
    Input('dropdown2', 'value'),
    prevent_initial_call=True
)
def start_order_2(value):
    # start_hub = list_actions[0]
    # final_hub = list_actions[-1]
    global taken_steps
    taken_steps = []

    # df_hubs['action'][start_hub]='start'
    # df_hubs['action'][final_hub]='final'
    global test_id
    if(value =='Test 1'):
        test_id = 0
    else:
        if(value == 'Test 2'):
            test_id = 1
        else:
            if(value == 'Test 3'):
                test_id = 2


    start_hub = df_test['Hubs'][test_id][0] #list_actions[0] 
    final_hub = df_test['Hubs'][test_id][-1] #list_actions[-1]

    # manipulate hubs dataframe based on start and final hub of test case
    df_hubs['action'][start_hub] = 'start'
    df_hubs['action'][final_hub] = 'final'

    env_config = {'pickup_hub_index': df_test['Pickup Hub'][test_id],
                      'delivery_hub_index': df_test['Delivery Hub'][test_id],
                      'pickup_timestamp': df_test['Pickup Time'][test_id],
                      'delivery_timestamp': df_test['Delivery Time'][test_id],
                      }
                      
    with open('env_config.pkl', 'wb') as f:
        pickle.dump(env_config, f)
    
    env.reset()
    print('-----',env.start_hub, env.final_hub, '------')

    print('Start, Final: ',env.start_hub, env.final_hub, env.position)
    print('Trips: ', env.available_actions)

    # nr_wait = 0
    # nr_shared = 0
    # nr_book = 0
    route_string='Calculated route: '

    # for i in type_of_actions:
    #     if i == 'wait':
    #         nr_wait+=1
    #     if i == 'share':
    #         nr_shared+=1
    #     if i == 'book':
    #         nr_book+=1

    for i in range(len(df_test['Hubs'][test_id])):
        route_string += str(df_test['Hubs'][test_id][i]) + ' ->'
    route_string = route_string[0:-3]


    return html.Div('CURRENT ORDER: {} -> {}'.format(start_hub,final_hub)), route_string



if __name__ == '__main__':
    app.run_server()