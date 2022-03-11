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

    def __init__(self, graph, start_hub, final_hub): # TODO: add action space -> import gym.spaces -> action_space = Discrete(3)
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

    def makeMove(self, action):
        #TODO
        #need to determine what object (if any) is in the new grid spot the player is moving to
        #actions in {u,d,l,r}
        oldposition = self.position

        if action == 'u': #up
            nxtposition = (self.position[0] - 1, self.position[1])
        elif action == 'd': #down
            nxtposition = (self.position[0] + 1, self.position[1])
        elif action == 'l': #left
            nxtposition = (self.position[0], self.position[1] - 1)
        elif action == 'r': #right
            nxtposition = (self.position[0], self.position[1] + 1)
        else:
            pass

        if (nxtposition[0] >= 0) and (nxtposition[0] <= (self.board_rows -1)):
            if (nxtposition[1] >= 0) and (nxtposition[1] <= (self.board_cols -1)):
                if nxtposition != (1, 1):
                    self.position = nxtposition

        return self.position, self.reward(), self.isDone()

    def isDone(self):
        return self.position == self.final_hub
    
    def reward(self): # TODO: extend function: should not return 0 reward if position is a second time on start_hub
        if self.isDone():
            return 10
        elif self.position == self.start_hub: 
            return 0
        else:
            return -1

    def visualize(self):
        current_pos_x = self.graph.nodes[self.position]['x']
        current_pos_y = self.graph.nodes[self.position]['y']
        final_hub_x = self.graph.nodes[self.final_hub]['x']
        final_hub_y = self.graph.nodes[self.final_hub]['y']
        start_hub_x = self.graph.nodes[self.start_hub]['x']
        start_hub_y = self.graph.nodes[self.start_hub]['y']

        # Create plot
        plot = ox.plot_graph_folium(self.graph,fit_bounds=True, weight=2, color="#333333")

        # Place markers for start, final and current position
        folium.Marker(location=[final_hub_y, final_hub_x], icon=folium.Icon(color='gray', prefix='fa', icon='flag-checkered')).add_to(plot)
        folium.Marker(location=[current_pos_y, current_pos_x], icon=folium.Icon(color='lightgreen', prefix='fa',icon='cube')).add_to(plot)
        folium.Marker(location=[start_hub_y, start_hub_x], icon=folium.Icon(color='beige', prefix='fa', icon='caret-right')).add_to(plot)
        
        # Plot
        pos_to_final = nx.shortest_path(self.graph, self.position, self.final_hub, weight="travel_time")
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