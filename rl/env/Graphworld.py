import numpy as np
import StreetGraph
import osmnx as ox
import networkx as nx
import folium
from folium.plugins import MarkerCluster

# global variables
#board_rows = 3
#board_colsboard_cols = 4

#LOSE_position = (1, 3)
#start_hub = (2, 0)
#DETERMINISTIC = True

class Environment:

    def __init__(self, graph: nx.MultiDiGraph, start_hub: int, final_hub: int): # TODO: add action space -> import gym.spaces -> action_space = Discrete(3)
        """_summary_

        Args:
            graph (nx.MultiDiGraph): graph
            start_hub (int): nodeId
            final_hub (int): nodeId

        """

        self.graph = graph.graph

        if self.graph.has_node(start_hub):
            self.start_hub = start_hub
            self.position = start_hub
        else:
            return 'Initialized start hub was not found in graph'

        if self.graph.has_node(final_hub):
            self.final_hub = final_hub
        else:
            return 'Initialized final hub was not found in graph'

    def step(self, action):
        """ Executes an action based on the index passed as a parameter

        Args:
            action (int): index of action to be taken from availableActions

        Returns:
            int: new position
            int: new reward
            boolean: isDone
        """
        old_position = self.position
        neighbors = self.availableActions()

        if self.validateAction(action):
            self.position = neighbors[action]
        else:
            pass

        return self.position, self.reward(), self.isDone()

    def availableActions(self):
        neighbors = list(self.graph.neighbors(self.position))
        return neighbors

    def validateAction(self, action):
        return action < len(self.availableActions())

    def isDone(self):
        return self.position == self.final_hub
    
    def reward(self): # TODO: extend function: should not return 0 reward if position is a second time on start_hub
        if self.isDone():
            return 10
        elif self.position == self.start_hub: 
            return 0
        else:
            return -1

    def render(self, visualize_actionspace: bool = False):
        """_summary_

        Args:
            visualize_actionspace (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """
        current_pos_x = self.graph.nodes[self.position]['x']
        current_pos_y = self.graph.nodes[self.position]['y']
        final_hub_x = self.graph.nodes[self.final_hub]['x']
        final_hub_y = self.graph.nodes[self.final_hub]['y']
        start_hub_x = self.graph.nodes[self.start_hub]['x']
        start_hub_y = self.graph.nodes[self.start_hub]['y']

        # Create plot
        plot = ox.plot_graph_folium(self.graph,fit_bounds=True, weight=2, color="#333333")

        # Place markers for start, final and current position
        folium.Marker(location=[final_hub_y + 10, final_hub_x], icon=folium.Icon(color='red', prefix='fa', icon='flag-checkered')).add_to(plot)
        folium.Marker(location=[start_hub_y, start_hub_x], icon=folium.Icon(color='lightblue', prefix='fa', icon='caret-right')).add_to(plot)
        folium.Marker(location=[current_pos_y, current_pos_x], icon=folium.Icon(color='lightgreen', prefix='fa',icon='cube')).add_to(plot)

        if(visualize_actionspace):
            for i, target_node in enumerate(self.availableActions()):
                target_node_x = self.graph.nodes[target_node]['x']
                target_node_y = self.graph.nodes[target_node]['y']
                popup = "%s: go to node %d" % (i, target_node)
                folium.Marker(location=[target_node_y, target_node_x], popup = popup, tooltip=str(i)).add_to(plot)

        # Plot
        pos_to_final = nx.shortest_path(self.graph, self.position, self.final_hub, weight="travel_time")
        print(pos_to_final)
        if(not len(pos_to_final)< 2):
            ox.plot_route_folium(G=self.graph,route=pos_to_final,route_map=plot)

        return plot

    def reset(self):
        # self.isDone = False
        self.position = self.start_hub
        pass

# class Agent:

#     def __init__(self):
#         self.positions = []
#         self.actions = ["up", "down", "left", "right"]
#         self.position = position()
#         # self.lr = 0.2
#         # self.exp_rate = 0.3

#         # initial position self.reward
#         self.position_values = {}
#         for i in range(board_rows):
#             for j in range(board_colsboard_cols):
#                 self.position_values[(i, j)] = 0  # set initial value to 0

#     # def chooseAction(self):
#     #     # choose action with most expected value
#     #     mx_nxt_reward = 0
#     #     action = ""

#     #     if np.random.uniform(0, 1) <= self.exp_rate:
#     #         action = np.random.choice(self.actions)
#     #     else:
#     #         # greedy action
#     #         for a in self.actions:
#     #             # if the action is deterministic
#     #             nxt_reward = self.position_values[self.position.nxtPosition(a)]
#     #             if nxt_reward >= mx_nxt_reward:
#     #                 action = a
#     #                 mx_nxt_reward = nxt_reward
#     #     return action

#     def takeAction(self, action):
#         position = self.position.nxtPosition(action)
#         return position(position=position)

#     def reset(self):
#         self.positions = []
#         self.position = position()

#     def play(self, rounds=10):
#         i = 0
#         while i < rounds:
#             # to the end of game back propagate self.reward
#             if self.position.isEnd:
#                 # back propagate
#                 self.reward = self.position.giveReward()
#                 # explicitly assign end position to self.reward values
#                 self.position_values[self.position.position] = self.reward  # this is optional
#                 print("Game End Reward", self.reward)
#                 for s in reversed(self.positions):
#                     self.reward = self.position_values[s] + self.lr * (self.reward - self.position_values[s])
#                     self.position_values[s] = round(self.reward, 3)
#                 self.reset()
#                 i += 1
#             else:
#                 action = self.chooseAction()
#                 # append trace
#                 self.positions.append(self.position.nxtPosition(action))
#                 print("current position {} action {}".format(self.position.position, action))
#                 # by taking the action, it reaches the next position
#                 self.position = self.takeAction(action)
#                 # mark is end
#                 self.position.isEndFunc()
#                 print("nxt position", self.position.position)
#                 print("---------------------")

#     def showValues(self):
#         for i in range(0, board_rows):
#             print('----------------------------------')
#             out = '| '
#             for j in range(0,board_cols):
#                 out += str(self.position_values[(i, j)]).ljust(6) + ' | '
#             print(out)
#         print('----------------------------------')