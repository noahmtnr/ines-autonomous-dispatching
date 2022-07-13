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
from gym.spaces import Dict

import ray
from ray.rllib.agents.dqn import DQNTrainer, DEFAULT_CONFIG
from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense,LSTM,GRU,Dropout, Flatten
from ray.rllib.models.tf.fcnet import FullyConnectedNetwork
from ray.rllib.models.tf.misc import normc_initializer
from gym import spaces
from ray.rllib.utils import try_import_tf

sys.path.insert(0,"")

from Manhattan_Graph_Environment.graphs.ManhattanGraph import ManhattanGraph
from Manhattan_Graph_Environment.gym_graphenv.envs.GraphworldManhattan import GraphEnv, CustomCallbacks

wandb.login(key="93aab2bcc48447dd2e8f74124d0258be2bf93859")
wandb.init(project="HP-reduce_bookowns", entity="hitchhike")

tf_api, tf_original, tf_version = try_import_tf(error = True)

class ActionMaskModel(TFModelV2):
    """Model that handles simple discrete action masking.
    This assumes the outputs are logits for a single Categorical action dist.
    Getting this to work with a more complex output (e.g., if the action space
    is a tuple of several distributions) is also possible but left as an
    exercise to the reader.
    """

    def __init__(
        self, obs_space, action_space, num_outputs, model_config, name, **kwargs
    ):

        orig_space = getattr(obs_space, "original_space", obs_space)
        print("Original Space:", orig_space )
        print("Original Space Shape", orig_space.shape)
        # assert (
        #     isinstance(orig_space, Dict)
        #     and "action_mask" in orig_space.spaces
        #     and "state" in orig_space.spaces
        # )

        super().__init__(obs_space, action_space, num_outputs, model_config, name)

        self.internal_model = FullyConnectedNetwork(
            orig_space["observations"],
            action_space,
            num_outputs,
            model_config,
            name + "_internal",
        )

        # disable action masking --> will likely lead to invalid actions
        # self.no_masking = model_config["custom_model_config"].get("no_masking", False)

    def forward(self, input_dict, state, seq_lens):
        # Extract the available actions tensor from the observation.
        action_mask = input_dict["obs"]["action_mask"]

        # Compute the unmasked logits.
        logits, _ = self.internal_model({"obs": input_dict["obs"]["observations"]})

        
       

        # Convert action_mask into a [0.0 || -inf]-type mask.
        inf_mask = tf.maximum(tf.math.log(action_mask), tf.float32.min)
        masked_logits = logits + inf_mask

        # Return masked logits.
        return masked_logits, state

    def value_function(self):
        return self.internal_model.value_function()

ModelCatalog.register_custom_model("my_mask_model", ActionMaskModel)

env=GraphEnv()


# Initialize Ray
ray.init()


rainbow_config = DEFAULT_CONFIG.copy()
rainbow_config['num_workers'] = 3
rainbow_config["train_batch_size"] = 400
rainbow_config["gamma"] = 0.99
# rainbow_config["framework"] = "torch"
rainbow_config["callbacks"] = CustomCallbacks
#rainbow_config["hiddens"] = [70] # to try with 1024  //was also 516
rainbow_config["model"] = {
    "custom_model": ActionMaskModel,
    # "fcnet_hiddens": [70],
    # "fcnet_activation": "relu",
    # "custom_model_config": {"no_masking": False},
}

#num_gpus and other gpu parameters in order to train with gpu
#rainbow_config["num_gpus"] = int(os.environ.get("RLLIB_NUM_GPUS", "0"))

#rainbow parameters

# N-step Q learning
rainbow_config["n_step"]= 3 #[between 1 and 10]  //was 5 and 7
# Whether to use noisy network
rainbow_config["noisy"] = True
# rainbow_config["sigma0"] = 0.2
# Number of atoms for representing the distribution of return. When
# this is greater than 1, distributional Q-learning is used.
# the discrete supports are bounded by v_min and v_max
rainbow_config["num_atoms"] = 70 #[more than 1] //was 51,20
rainbow_config["v_min"] =-210000
rainbow_config["v_max"]=210000 # (set v_min and v_max according to your expected range of returns).


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
               'count_delivered_on_time': int(result["count_delivered_on_time"]),
               'count_delivered_with_delay': int(result["count_delivered_with_delay"]),
               'count_not_delivered': int(result["count_not_delivered"]),
                'share_delivered_on_time': float(result["count_delivered_on_time"]/result['episodes_this_iter'])
               }
    episode_data.append(episode)
    episode_json.append(json.dumps(episode))
    file_name = trainer.save(checkpoint_root)
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
                'count_delivered_on_time': result["count_delivered_on_time"],
                'count_delivered_with_delay': result["count_delivered_with_delay"],
                'count_not_delivered': result["count_not_delivered"],
                'share_delivered_on_time': result["count_delivered_on_time"]/result['episodes_this_iter']
    })

    print(f'{n + 1:3d}: Min/Mean/Max reward: {result["episode_reward_min"]:8.4f}/{result["episode_reward_mean"]:8.4f}/{result["episode_reward_max"]:8.4f}, len mean: {result["episode_len_mean"]:8.4f}. Checkpoint saved to {file_name}')