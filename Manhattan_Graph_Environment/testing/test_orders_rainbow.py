"""
Test File for creating orders and testing the rainbow agent on the orders.
"""

# imports
import csv
import sys

sys.path.insert(0, "")
import os

os.environ['HDF5_DISABLE_VERSION_CHECK'] = '2'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import pickle
import warnings

warnings.filterwarnings('ignore')
from ray.rllib.agents.dqn import DQNTrainer, DEFAULT_CONFIG
from config.definitions import ROOT_DIR
from Manhattan_Graph_Environment.gym_graphenv.envs.GraphworldManhattan import GraphEnv, CustomCallbacks

sys.path.append(os.path.join(ROOT_DIR, "Manhattan_Graph_Environment", "gym_graphenv"))


# class definition
class TestOrders:
    """
    Init Method of Class
    """

    def __init__(self, ):
        sys.path.insert(0, "")
        # Set trainer configuration
        self.trainer_config = DEFAULT_CONFIG.copy()
        self.trainer_config['num_workers'] = 1
        self.trainer_config["train_batch_size"] = 400
        self.trainer_config["gamma"] = 0.99
        # rainbow_config["framework"] = "torch"
        self.trainer_config["callbacks"] = CustomCallbacks
        self.trainer_config["hiddens"] = [180, 150, 100]  # to try with 1024  //was also 516
        self.trainer_config["model"] = {
            # "custom_model": "my_tf_model",
            "fcnet_activation": 'relu',
        }

        # num_gpus and other gpu parameters in order to train with gpu
        # self.trainer_config["num_gpus"] = int(os.environ.get("RLLIB_NUM_GPUS", "0"))

        # rainbow parameters

        # N-step Q learning
        self.trainer_config["n_step"] = 4  # [between 1 and 10]  //was 5 and 7
        # Whether to use noisy network
        self.trainer_config["noisy"] = True
        # rainbow_config["sigma0"] = 0.2
        # Number of atoms for representing the distribution of return. When
        # this is greater than 1, distributional Q-learning is used.
        # the discrete supports are bounded by v_min and v_max
        self.trainer_config["num_atoms"] = 70  # [more than 1] //was 51,20
        self.trainer_config["v_min"] = -15000
        self.trainer_config["v_max"] = 15000  # (set v_min and v_max according to your expected range of returns).

        # here from trainRainbow die config
        # self.trainer_config["train_batch_size"] = 400
        # self.trainer_config["framework"] = "torch"

    """
    Run the agent on the orders.
    """

    def test_order(self):

        # Initialize trainer
        rainbow_trainer = DQNTrainer(self.trainer_config, GraphEnv)
        # checkpoint anpassen
        file_name = os.path.join(ROOT_DIR, 'tmp', 'rainbow-new', 'rllib_checkpoint', 'checkpoint_000081',
                                 'checkpoint-81')
        print(file_name)

        # Restore the Trainer
        rainbow_trainer.restore(file_name)
        # env = gym.make('graphworld-v0')
        env = GraphEnv(use_config=True)
        obs = env.reset()
        print(env.position)
        print("reset done")

        # get information
        list_nodes = []
        list_hubs = [env.position]
        list_actions = ["start"]
        rem_dist = [env.learn_graph.adjacency_matrix('remaining_distance')[env.position][env.final_hub]]
        # route = [env_config["pickup_hub_index"]]
        # route_timestamps = [datetime.strptime(env_config["pickup_timestamp"], '%Y-%m-%d %H:%M:%S')]
        sum_reward = 0
        # sum_travel_time = timedelta(seconds=0)
        # print(sum_travel_time)
        # sum_distance = 0
        # results = []
        done = False
        while not done:
            action = rainbow_trainer.compute_action(obs)
            print("action test", action)
            state, reward, done, info = env.step(action)
            sum_reward += reward

            if (info["route"][-1] != env.final_hub and info["action"] != "Wait"):
                print(list_nodes)
                list_nodes.extend(info["route"][0:-1])
                print(list_nodes)
            list_hubs.append(info["hub_index"])
            list_actions.append(info["action"])
            rem_dist.append(info["remaining_dist"])

            # get data from action
            # route.append(action)
            # route_timestamps.append(info.get('timestamp'))

            # sum_travel_time += timedelta(seconds=info.get('step_travel_time'))
            # delivey_time = datetime.strptime(env_config["delivery_timestamp"], '%Y-%m-%d %H:%M:%S')
            # time_until_deadline = delivey_time - sum_travel_time
            # sum_distance += info.get('distance') / 1000
            # number_hubs = info.get('count_hubs')
            # # add reward
            # sum_reward += reward

            # check if finished
            if done == True:
                print("DELIVERY DONE! sum_reward: ", sum_reward)
                print("DELIVERY DONE! Route: ", list_hubs)
                print("DELIVERY DONE! Actions: ", list_actions)
                print("DELIVERY DONE! Nodes: ", list_nodes)
                break

            # print("sum_reward: ",sum_reward)
            # print("sum_reward: ",sum_reward, " time: ",env.time, "deadline time: ", env.deadline, "pickup time: ", env.pickup_time)
        # reward_list = {"pickup_hub": env_config['pickup_hub_index'], "delivery_hub": env_config['delivery_hub_index'],
        #                "reward": sum_reward, "hubs": number_hubs, "route": route, "time": str(sum_travel_time),
        #                "dist": sum_distance, "time_until_deadline": time_until_deadline, "timestamps": route_timestamps}
        return list_hubs, list_actions, list_nodes, rem_dist


"""
Create one test order.
"""


def create_test_order():
    pickup_hub = 21
    delivery_hub = 5
    pickup_timestamp = "2016-01-05 20:46:00"
    delivery_timestamp = "2016-01-06 08:46:00"
    env_config = {'pickup_hub_index': pickup_hub,
                  'delivery_hub_index': delivery_hub,
                  'pickup_timestamp': pickup_timestamp,
                  'delivery_timestamp': delivery_timestamp,
                  }
    filepath = os.path.join(ROOT_DIR, 'env_config.pkl')
    with open(filepath, 'wb') as f:
        pickle.dump(env_config, f)

    test = TestOrders()
    list_hubs, list_actions, list_nodes, rem_dist = test.test_order()
    write_in_file_orders(list_hubs, list_actions, list_nodes, pickup_hub, delivery_hub, pickup_timestamp,
                         delivery_timestamp, rem_dist)


"""
Write a new order into a file with orders.
"""


def write_in_file_orders(hubs, actions, nodes, pickup_hub, delivery_hub, pickup_timestamp, delivery_timestamp,
                         rem_dist):
    filepath = os.path.join(ROOT_DIR, 'data', 'others', 'test_orders_dashboard.csv')
    mycsv = csv.reader(open(filepath))

    for row in mycsv:
        id = row[0]
    try:
        idInt = int(id)
    except:
        idInt = 0
    idInt += 1
    row_list = [
        [idInt, pickup_hub, delivery_hub, pickup_timestamp, delivery_timestamp, hubs, actions, nodes, rem_dist], ]

    with open(filepath, 'a+', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(row_list)


# call the method
create_test_order()
