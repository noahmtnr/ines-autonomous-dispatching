import numpy as np

# global variables
BOARD_ROWS = 3
BOARD_COLS = 4
final_hub = (0, 3)
#LOSE_STATE = (1, 3)
start_hub = (1, 0)
DETERMINISTIC = True


class State:
    def __init__(self, state=start_hub):
        self.board = np.zeros([BOARD_ROWS, BOARD_COLS])
        self.board[0,3] = 2
        self.state = state
        self.isEnd = False
        self.determine = DETERMINISTIC

    # def giveReward(self):
    #     if self.state == final_hub:
    #         return 1
    #     elif self.state == LOSE_STATE:
    #         return -1
    #     else:
    #         return 0

    def isEndFunc(self):
        if (self.state == final_hub):
            self.isEnd = True

    def nxtPosition(self, action):
        """
        action: up, down, left, right
        -------------
        0 | 1 | 2| 3|
        1 |
        2 |
        return next position
        """
        if self.determine:
            if action == "up":
                nxtState = (self.state[0] - 1, self.state[1])
            elif action == "down":
                nxtState = (self.state[0] + 1, self.state[1])
            elif action == "left":
                nxtState = (self.state[0], self.state[1] - 1)
            else:
                nxtState = (self.state[0], self.state[1] + 1)
            # if next state legal
            if (nxtState[0] >= 0) and (nxtState[0] <= (BOARD_ROWS -1)):
                if (nxtState[1] >= 0) and (nxtState[1] <= (BOARD_COLS -1)):
                    if nxtState != (1, 1):
                        return nxtState
            return self.state

    def showBoard(self):
        self.board[self.state] = 1
        #self.board[final_hub] = 2
        for i in range(0, BOARD_ROWS):
            print('-----------------')
            out = '| '
            for j in range(0, BOARD_COLS):
                if self.board[i, j] == 1:
                    token = '*'
                if self.board[i, j] == 2:
                    token = 'F'
                if self.board[i, j] == 0:
                    token = '0'
                out += token + ' | '
            print(out)
        print('-----------------')


