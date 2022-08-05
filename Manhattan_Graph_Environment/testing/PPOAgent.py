"""
PPO (Proximal Policy Optimization) Agent.
"""

# imports

import sys
sys.path.insert(0,"")
# print("System path",sys.path)

import os
from datetime import datetime, timedelta
from config.definitions import ROOT_DIR
import warnings
warnings.filterwarnings('ignore')
from ray.rllib.agents.ppo import PPOTrainer, DEFAULT_CONFIG
from Manhattan_Graph_Environment.gym_graphenv.envs.GraphworldManhattan import GraphEnv

sys.path.append(os.path.join(ROOT_DIR, "Manhattan_Graph_Environment", "gym_graphenv"))


# class definition
class PPOAgent:
    """
    Init Method of Class
    : param env: Environment Object
    """
    def __init__(self, env):
        sys.path.insert(0,"")
        #Set trainer configuration
        self.trainer_config = DEFAULT_CONFIG.copy()
        # self.trainer_config["train_batch_size"] = 400
        # self.trainer_config["framework"] = "torch"
        self.env = env

    """
    Runs the agent in the environment (by taking steps according to policy of agent) until it reaches the final hub.
    :param env: 
    :param reward_list:
    :param env_config:
    :return: dictionary containing results of run.
    """
    def run_one_episode (self,reward_list,env_config):   
        # Initialize trainer
        ppo_trainer=PPOTrainer(self.trainer_config,env=self.env)
        mode={ "env-name":"graphworld-v0",
        "env":GraphEnv,
        "iterations":1,
        }
        # checkpoint anpassen
        file_name = os.path.join(ROOT_DIR, 'tmp', 'ppo', 'graphworld', 'checkpoint_000010', 'checkpoint-10')

        #Restore the Trainer
        ppo_trainer.restore(file_name)
        env = self.env
        state = env.reset()
        print("reset done")

        # get information
        route=[env_config["pickup_hub_index"]]
        route_timestamps = [datetime.strptime(env_config["pickup_timestamp"], '%Y-%m-%d %H:%M:%S')]
        sum_reward = 0
        sum_travel_time = timedelta(seconds=0)
        print(sum_travel_time)
        sum_distance = 0
        results = []

        count_shares = 0
        count_bookowns = 0
        count_wait = 0
        steps = 0

        done = False
        # run until finished
        while not done:
            # take some action
            action = ppo_trainer.compute_action(state)
            state, reward, done, info = env.step(action)
            sum_reward += reward
            #env.render()

            # get information from action
            route.append(action)
            route_timestamps.append(info.get('timestamp'))

            
            sum_travel_time +=timedelta(seconds=info.get('step_travel_time'))
            delivey_time = datetime.strptime(env_config["delivery_timestamp"], '%Y-%m-%d %H:%M:%S')
            time_until_deadline= delivey_time-sum_travel_time
            sum_distance += info.get('distance')/1000
            number_hubs=info.get('count_hubs')
            dist_shares = info.get("dist_covered_shares")
            dist_bookowns = info.get("dist_covered_bookown")
            # add reward
            sum_reward += reward

            action_choice = info.get("action")

            if action_choice == "Share":
                count_shares += 1
            elif action_choice == "Book":
                count_bookowns += 1
            elif action_choice == "Wait":
                count_wait += 1
            steps += 1
            
            # check if finished
            if done == True:
                print("DELIVERY DONE! sum_reward: ",sum_reward)
                print("DELIVERY DONE! Route: ",route)
                print("DELIVERY DONE! Travel Time: ",sum_travel_time)
                print("DELIVERY DONE! Distance: ",sum_distance)
                print("DELIVERY DONE! Hubs: ",number_hubs)
                print("DELIVERY DONE! unitl deadline: ",time_until_deadline)

                # print("cumulative reward", sum_reward)
                # state = env.reset()
                # sum_reward = 0

            # print("sum_reward: ",sum_reward)
            # print("sum_reward: ",sum_reward, " time: ",env.time, "deadline time: ", env.deadline, "pickup time: ", env.pickup_time)
        if count_bookowns == 0:
            ratio = 0
        else:
            ratio = float(count_shares/count_bookowns)
        
        # results of the agent's run
        reward_list={"pickup_hub":env_config['pickup_hub_index'],"delivery_hub":env_config['delivery_hub_index'],"reward":sum_reward, "hubs":number_hubs, "route":route, "time":sum_travel_time, "dist":sum_distance, "time_until_deadline":time_until_deadline, "timestamps":route_timestamps, "count_bookowns": count_bookowns, "steps": steps, "ratio_share_to_own": ratio,"dist_covered_shares": dist_shares, "dist_covered_bookown": dist_bookowns}

        return reward_list