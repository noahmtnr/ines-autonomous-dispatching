sys.path.insert(0,"")

from Manhattan_Graph_Environment.graphs.ManhattanGraph import ManhattanGraph
from Manhattan_Graph_Environment.gym_graphenv.envs.GraphworldManhattan import GraphEnv, CustomCallbacks


env=GraphEnv()


env.reset()
print("Position: ", env.position)
action = 0
print("Action: ", action)
state, reward, done, info = env.step(action)
sum_reward+=reward
