import numpy as np
import pandas as pd
import json
import os
import shutil
import sys
import gym
import wandb

# CHANGES HERE
# uncomment if error appears
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
# CHANGES END HERE


import ray
from ray.rllib.agents.dqn import DQNTrainer, DEFAULT_CONFIG
from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense,LSTM,GRU,Dropout, Flatten
from ray.rllib.models.tf.misc import normc_initializer

sys.path.insert(0,"")

from Manhattan_Graph_Environment.graphs.ManhattanGraph import ManhattanGraph
from Manhattan_Graph_Environment.gym_graphenv.envs.GraphworldManhattan import GraphEnv, CustomCallbacks

wandb.login(key="93aab2bcc48447dd2e8f74124d0258be2bf93859")
wandb.init(project="Comparison_Hypothese_Normalisiert", entity="hitchhike")
# wandb.init(project="New_Metrics_Total_Env", entity="hitchhike")

# class CustomModel(TFModelV2):
#     def __init__(self, obs_space, action_space, num_outputs, model_config, name):
#         super(CustomModel, self).__init__(obs_space, action_space, num_outputs, model_config, name)
#         self.inputs = tf.keras.layers.Input(shape=obs_space.shape, name="final_hub")
#         layer_1 = tf.keras.layers.Dense(
#             600,
#             name="my_layer1",
#             activation=tf.nn.relu,
#             kernel_initializer=normc_initializer(1.0),
#         )(self.inputs)
#         layer_2 = tf.keras.layers.Dense(
#             500,
#             name="my_layer2",
#             activation=tf.nn.relu,
#             kernel_initializer=normc_initializer(1.0),
#         )(layer_1)
#         layer_3 = tf.keras.layers.Dense(
#             400,
#             name="my_layer3",
#             activation=tf.nn.relu,
#             kernel_initializer=normc_initializer(1.0),
#         )(layer_2)
#         layer_4 = tf.keras.layers.Dense(
#             300,
#             name="my_layer4",
#             activation=tf.nn.relu,
#             kernel_initializer=normc_initializer(1.0),
#         )(layer_3)
#         layer_5 = tf.keras.layers.Dense(
#             120,
#             name="my_layer5",
#             activation=tf.nn.relu,
#             kernel_initializer=normc_initializer(1.0),
#         )(layer_4)
#         layer_out = tf.keras.layers.Dense(
#             num_outputs,
#             name="my_out",
#             activation=None,
#             kernel_initializer=normc_initializer(0.01),
#         )(layer_5)
#         value_out = tf.keras.layers.Dense(
#             1,
#             name="value_out",
#             activation=None,
#             kernel_initializer=normc_initializer(0.01),
#         )(layer_5)
#         self.base_model = tf.keras.Model(self.inputs, [layer_out, value_out])

#     def forward(self, input_dict, state, seq_lens):
#         model_out, self._value_out = self.base_model(input_dict["obs"])
#         return model_out, state

#     def value_function(self):
#         return tf.reshape(self._value_out, [-1])

# ModelCatalog.register_custom_model("my_tf_model", CustomModel)


# Initialize Ray
ray.init()


rainbow_config = DEFAULT_CONFIG.copy()
rainbow_config['num_workers'] = 1
rainbow_config["train_batch_size"] = 400
rainbow_config["gamma"] = 0.99
# rainbow_config["framework"] = "torch"
rainbow_config["callbacks"] = CustomCallbacks
rainbow_config["hiddens"] = [180,150,100] # to try with 1024  //was also 516
rainbow_config["model"] = {
    # "custom_model": "my_tf_model",
    "fcnet_activation": 'relu',
}

#num_gpus and other gpu parameters in order to train with gpu
rainbow_config["num_gpus"] = int(os.environ.get("RLLIB_NUM_GPUS", "0"))

#rainbow parameters

# N-step Q learning
rainbow_config["n_step"]= 4 #[between 1 and 10]  //was 5 and 7
# Whether to use noisy network
rainbow_config["noisy"] = True
# rainbow_config["sigma0"] = 0.2
# Number of atoms for representing the distribution of return. When
# this is greater than 1, distributional Q-learning is used.
# the discrete supports are bounded by v_min and v_max
rainbow_config["num_atoms"] = 70 #[more than 1] //was 51,20
rainbow_config["v_min"] =-15000
rainbow_config["v_max"]=15000# (set v_min and v_max according to your expected range of returns).


# Initialize trainer
trainer = DQNTrainer(rainbow_config,GraphEnv )

#Define the path where the results of the trainer should be saved
checkpoint_root = "tmp/rainbow/graphworld"
shutil.rmtree(checkpoint_root, ignore_errors=True, onerror=None)   # clean up old runs
ray_results = "{}/ray_results/".format(os.getenv("HOME"))
shutil.rmtree(ray_results, ignore_errors=True, onerror=None)   # clean up old runs


# Run trainer

results = []
episode_data = []
episode_json = []
n_iter = 100
for n in range(n_iter):
    result = trainer.train()
    results.append(result)
    print("Episode", n)
    print("Result",result)
    episode = {'n': n,
               'n_trained_episodes': int(result['episodes_this_iter']),
               'episode_reward_min': float(result['episode_reward_min']),
               'episode_reward_mean': float(result['episode_reward_mean']),
               'episode_reward_max': float(result['episode_reward_max']),
               'episode_len_mean': float(result['episode_len_mean']),
               'count_wait_min': int(result["count_wait_min"]),
               'count_wait_max': int(result["count_wait_max"]),
               'count_wait_mean': float(result["count_wait_mean"]),
               'count_bookown_min': int(result["count_bookown_min"]),
               'count_bookown_max': int(result["count_bookown_max"]),
               'count_bookown_mean': float(result["count_bookown_mean"]),
               'count_share_min': int(result["count_share_min"]),
               'count_share_max': int(result["count_share_max"]),
               'count_share_mean': float(result["count_share_mean"]),
               'share_wait_min': float(result["share_wait_min"]),
               'share_wait_max': float(result["share_wait_max"]),
               'share_wait_mean': float(result["share_wait_mean"]),
               'share_bookown_min': float(result["share_bookown_min"]),
               'share_bookown_max': float(result["share_bookown_max"]),
               'share_bookown_mean': float(result["share_bookown_mean"]),
               'share_share_min': float(result["share_share_min"]),
               'share_share_max': float(result["share_share_max"]),
               'share_share_mean': float(result["share_share_mean"]),
               'share_to_own_ratio_min': float(result["share_to_own_ratio_min"]),
               'share_to_own_ratio_max': float(result["share_to_own_ratio_max"]),
               'share_to_own_ratio_mean': float(result["share_to_own_ratio_mean"]),
               'count_steps_min': float(result["count_steps_min"]),
               'count_steps_max': float(result["count_steps_max"]),
               'count_steps_mean': float(result["count_steps_mean"]),
               'count_terminated': int(result["count_terminated"]),
               'count_delivered_on_time': int(result["count_delivered_on_time"]),
               'count_delivered_with_delay': int(result["count_delivered_with_delay"]),
               'count_not_delivered': int(result["count_not_delivered"]),
                'share_delivered_on_time': float(result["count_delivered_on_time"]/result['episodes_this_iter']),

               # new metrics
               'boolean_has_booked_any_own': int(result["boolean_has_booked_any_own"]),
               # 'bookowns_to_all': float(result["bookowns_to_all"]),
               'count_shared_available': int(result["count_shared_available"]),
               "ratio_shared_available_to_all_steps": float(result["ratio_shared_available_to_all_steps"]),
               'count_shared_available_useful': int(result["count_shared_available_useful"]),
               'shared_taken_to_shared_available': float(result["shared_taken_to_shared_available"]),
               'shared_available_useful_to_shared_available': float(result["shared_available_useful_to_shared_available"]),
               'shared_taken_useful_to_shared_available_useful': float(result["shared_taken_useful_to_shared_available_useful"]),

               # share delivered without bookown von allen delivered
               'ratio_delivered_without_bookown_to_all_delivered': float(result["ratio_delivered_without_bookown_to_all_delivered"]),
               # share delivered without bookown oder von allen
               # Anteil des Weges den man hätte durch shared rides abdecken können + den den wir tatsächlich mit shares abgedeckt haben

               #new metrics bookown distance reduced and rem distnce reduced
                'bookown_distance_not_covered_share':float(result['bookown_distance_not_covered_share']),
                'bookown_distance_not_covered': float(result['bookown_distance_not_covered']),
                'distance_reduced_with_ownrides':float(result['distance_reduced_with_ownrides']),
                'distance_reduced_with_shared':float(result['distance_reduced_with_shared']),
                'distance_reduced_with_ownrides_share':float(result['distance_reduced_with_ownrides_share']),
                'distance_reduced_with_shared_share':float(result['distance_reduced_with_shared_share']),


               }
    episode_data.append(episode)
    episode_json.append(json.dumps(episode))
    file_name = trainer.save(checkpoint_root)
    trainer.save(os.path.join(wandb.run.dir, "checkpoint"))
    # wandb.save(file_name)
    wandb.log({"n_trained_episodes": result['episodes_this_iter'],
                "mean_reward": result['episode_reward_mean'],
                "max_reward": result['episode_reward_max'],
                "own_mean": result['count_bookown_mean'],
                "wait_mean": result['count_wait_mean'],
                "share_mean": result['count_share_mean'],
                "share_of_bookown_mean": result['share_bookown_mean'],
                "share_of_wait_mean": result['share_wait_mean'],
                "share_of_share_mean": result['share_share_mean'],
                "share_to_own_ratio_max": result['share_to_own_ratio_max'],
                "share_to_own_ratio_mean": result['share_to_own_ratio_mean'],
                'count_steps_mean': result["count_steps_mean"],
                'count_terminated': result["count_terminated"],
                'count_delivered_on_time': result["count_delivered_on_time"],
                'count_delivered_with_delay': result["count_delivered_with_delay"],
                'count_not_delivered': result["count_not_delivered"],
                'share_delivered_on_time': result["count_delivered_on_time"]/result['episodes_this_iter'],

                # new metrics
                'boolean_has_booked_any_own': result["boolean_has_booked_any_own"],
                #'bookowns_to_all': result["bookowns_to_all"],
                'count_shared_available': result["count_shared_available"],
                'count_shared_available_useful': result["count_shared_available_useful"],
                'shared_taken_to_shared_available': result["shared_taken_to_shared_available"],
                'shared_available_useful_to_shared_available': result["shared_available_useful_to_shared_available"],
                'shared_taken_useful_to_shared_available_useful': result["shared_taken_useful_to_shared_available_useful"],
                "ratio_shared_available_to_all_steps": result["ratio_shared_available_to_all_steps"],

                "ratio_delivered_without_bookown_to_all_delivered": result["ratio_delivered_without_bookown_to_all_delivered"],


                'bookown_distance_not_covered_share': result['bookown_distance_not_covered_share'],
                'bookown_distance_not_covered': result['bookown_distance_not_covered'],
                'distance_reduced_with_ownrides':result['distance_reduced_with_ownrides'],
                'distance_reduced_with_shared':result['distance_reduced_with_shared'],
                'distance_reduced_with_ownrides_share':result['distance_reduced_with_ownrides_share'],
                'distance_reduced_with_shared_share':result['distance_reduced_with_shared_share'],

    })

    print(f'{n + 1:3d}: Min/Mean/Max reward: {result["episode_reward_min"]:8.4f}/{result["episode_reward_mean"]:8.4f}/{result["episode_reward_max"]:8.4f}, len mean: {result["episode_len_mean"]:8.4f}. Checkpoint saved to {file_name}')