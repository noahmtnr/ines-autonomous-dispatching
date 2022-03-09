import numpy as np

# global variables
#board_rows = 3
#board_colsboard_cols = 4

#LOSE_position = (1, 3)
#start_hub = (2, 0)
#DETERMINISTIC = True

class Environment:

    def __init__(self, position, final_hub, board_rows, board_cols): # TODO: add action space -> import gym.spaces -> action_space = Discrete(3)
        # self.board = np.zeros([board_rows,board_cols])
        # self.board[final_hub] = 2
        # self.board[position] = 1
        #self.reward = 0
        self.start_hub = position
        self.position = position
        self.final_hub = final_hub
        self.board_rows = board_rows
        self.board_cols = board_cols
        #self.determine = DETERMINISTIC


    # def isEndFunc(self):
    #     if (self.position == self.final_hub):
    #         self.isEnd = True


    def makeMove(self, action):
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
        # TODO: extend to visualize both agent position and start/final hub even if equal
        self.board = np.zeros([self.board_rows,self.board_cols])
        self.board[self.start_hub] = 1
        self.board[self.final_hub] = 2
        self.board[self.position] = 3

        for i in range(0, self.board_rows):
            print('-----------------')
            out = '| '
            for j in range(0,self.board_cols):
                if self.board[i, j] == 1:
                    token = 'S'
                if self.board[i, j] == 2:
                    token = 'F'
                if self.board[i, j] == 3:
                    token = '*'
                if self.board[i, j] == 0:
                    token = '0'
                out += token + ' | '
            print(out)
        print('-----------------')

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