"""
Wrapper Class for Abstraction of Benchmark and RL Agents
Methods: file_read, read_orders, proceed_order
"""

# imports
import warnings

warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=UserWarning)
import sys
import os

sys.path.insert(0, "")
from config.definitions import ROOT_DIR

sys.path.append(os.path.join(ROOT_DIR, "Manhattan_Graph_Environment", "gym_graphenv"))
from RandomAgent import RandomAgent
from CostAgent import CostAgent
from SharesAgent import SharesAgent
from RainbowAgent import RainbowAgent
from PPOAgent import PPOAgent
from DQNAgent import DQNAgent
from BookownAgent import BookownAgent
from SharesBookEndAgent import SharesBookEndAgent
import pandas as pd
import pickle


# class definition
class BenchmarkWrapper:
    """
    Init Method of Class
    :param agent_name: String
        can be random, cost, Bookown, SharesBookEnd for a Benchmark Agent
        can be DQN, PPO, rainbow for an RL Agent
    : param env: Environment Object
    """

    def __init__(self, agent_name, env):
        if (agent_name != None):
            self.name = agent_name
        else:
            self.name = sys.argv[2]
        self.env = env
        self.manhattan_graph = self.env.get_Graph()

    """
    Reads the first X lines of an orders file.
    The number of lines can be changed by adapting nrows when reading csv.
    Method has no parameters.
    :return: Pandas DataFrame containing orders
    """

    def file_read(self):

        # use if you want to test randomly generated orders which are saved in random_orders csv file
        """
        # for testing random orders
        if len(sys.argv) > 1:
            first_arg = sys.argv[1]
            file_path = os.path.join(ROOT_DIR, "data", "others", 'random_orders.csv') + first_arg
            orders = pd.read_csv(file_path, nrows=1)
        else:
            filepath = os.path.join(ROOT_DIR, "data", "others", 'random_orders.csv')
            orders = pd.read_csv(filepath, nrows=1)
        """

        # use if you want to test specific orders which are saved in test_orders csv file
        # for testing specific test orders
        if len(sys.argv) > 1:
            first_arg = sys.argv[1]
            file_path = os.path.join(ROOT_DIR, "data", "others", 'test_orders.csv') + first_arg
            orders = pd.read_csv(file_path, nrows=11)
        else:
            filepath = os.path.join(ROOT_DIR, "data", "others", 'test_orders.csv')
            orders = pd.read_csv(filepath, nrows=11)

        return orders

    """
    Reads the lines of the file and itemizes it into nodes and timestamps of the order.
    Method has no parameters.
    :return: Array containing the orders, each as dictionary.
    """

    def read_orders(self):
        reward_list = []
        orders = self.file_read()
        for index, row in orders.iterrows():
            order = {"pickup_node": row['pickup_node_id'], "delivery_node": row['delivery_node_id'],
                     "pickup_timestamp": row['pickup_timestamp'], "delivery_timestamp": row['delivery_timestamp']}
            reward_list.append(self.proceed_order(order))
        return reward_list

    """
    Initializes an Agent depending on the name given.
    :param order: dictionary containing the current order.
    :return: dictionary with results of testing of the agent.
    """

    def proceed_order(self, order):
        # output the current order
        print("Current Order: ", order)

        # configures the environment
        # manhattan_graph = ManhattanGraph(filename='simple',hubs=120)
        pick_up_hub_index = self.manhattan_graph.get_hub_index_by_nodeid(order.get('pickup_node'))
        delivery_hub_index = self.manhattan_graph.get_hub_index_by_nodeid(order.get('delivery_node'))
        # print(pick_up_hub_index,delivery_hub_index)
        env_config = {'pickup_hub_index': pick_up_hub_index,
                      'delivery_hub_index': delivery_hub_index,
                      'pickup_timestamp': order.get('pickup_timestamp'),
                      'delivery_timestamp': order.get('delivery_timestamp')
                      }

        with open('env_config.pkl', 'wb') as f:
            pickle.dump(env_config, f)

        reward_list = []

        # selects an agent depending on the name
        for i in range(1):
            if self.name == "random":
                print("random")
                reward_list = RandomAgent.run_one_episode(self.env, reward_list, env_config)
            elif self.name == "cost":
                print("cost")
                reward_list = CostAgent.run_one_episode(self.env, reward_list, env_config)
            elif self.name == "DQN":
                print("DQN")
                dqn_Agent = DQNAgent(self.env)
                reward_list = dqn_Agent.run_one_episode(reward_list, env_config)
            elif self.name == "PPO":
                print("PPO")
                ppo_Agent = PPOAgent(self.env)
                reward_list = ppo_Agent.run_one_episode(reward_list, env_config)
            elif self.name == "Rainbow":
                print("Rainbow")
                Rainbow_Agent = RainbowAgent(self.env)
                reward_list = Rainbow_Agent.run_one_episode(reward_list, env_config)
            elif self.name == "Shares":
                print("Shares")
                reward_list = SharesAgent.run_one_episode(self.env, reward_list, env_config)
            elif self.name == "Bookown":
                print("Bookown")
                reward_list = BookownAgent.run_one_episode(self.env, reward_list, env_config)
            elif self.name == "SharesBookEnd":
                print("Shares with Book Own at End")
                reward_list = SharesBookEndAgent.run_one_episode(self.env, reward_list, env_config)
        return reward_list


"""
Main-Method for testing the BenchmarkWrapper.
"""
"""
def main():
    # benchmark = BenchmarkWrapper("random")
    # results = benchmark.read_orders()
    # print("Random",results)
    # benchmark2 = BenchmarkWrapper("cost")
    # results = benchmark2.read_orders()
    # print("Cost",results)
    # benchmark3 = BenchmarkWrapper("Rainbow")
    # DQNAgent()
    # results = benchmark3.read_orders()
    # print("Rainbow", results)
    benchmark = BenchmarkWrapper("Shares")
    results = benchmark.read_orders()
    print("SharesAgent: ", results)


main()
"""
