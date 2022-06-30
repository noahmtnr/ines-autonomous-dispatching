import numpy as np
import pandas as pd
import json
import os
#import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import shutil
import sys
import gym
import wandb

import ray
from ray.rllib.agents.ppo import PPOTrainer, DEFAULT_CONFIG

sys.path.insert(0,"")

from Manhattan_Graph_Environment.graphs.ManhattanGraph import ManhattanGraph
from Manhattan_Graph_Environment.gym_graphenv.envs.GraphworldManhattan import GraphEnv, CustomCallbacks

wandb.login(key="93aab2bcc48447dd2e8f74124d0258be2bf93859")
wandb.init(project="Manhattan-ppo", entity="hitchhike")

env=GraphEnv()


# Initialize Ray
ray.init()

#Set trainer configuration
trainer_config = DEFAULT_CONFIG.copy()
# trainer_config['num_workers'] = 1
# trainer_config["train_batch_size"] = 400
# trainer_config["gamma"] = 0.95
# trainer_config["n_step"] = 10
trainer_config["framework"] = "torch"
trainer_config["callbacks"] = CustomCallbacks
trainer_config["explore"] = True
trainer_config["exploration_config"] = {
    "type": "EpsilonGreedy",
    "initial_epsilon": 1.0,
    "final_epsilon": 0.02,
}
#trainer_config["num_gpus"] = int(os.environ.get("RLLIB_NUM_GPUS", "0"))


# Initialize trainer
trainer = PPOTrainer(trainer_config,GraphEnv )

#Define the path where the results of the trainer should be saved
checkpoint_root = "tmp/ppo/graphworld"
shutil.rmtree(checkpoint_root, ignore_errors=True, onerror=None)   # clean up old runs
ray_results = "{}/ray_results/".format(os.getenv("HOME"))
shutil.rmtree(ray_results, ignore_errors=True, onerror=None)   # clean up old runs

# Run trainer

results = []
episode_data = []
episode_json = []
n_iter = 1

for n in range(n_iter):
    result = trainer.train()
    results.append(result)
    print("Episode", n)

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
    })

    print(f'{n + 1:3d}: Min/Mean/Max reward: {result["episode_reward_min"]:8.4f}/{result["episode_reward_mean"]:8.4f}/{result["episode_reward_max"]:8.4f}, len mean: {result["episode_len_mean"]:8.4f}. Checkpoint saved to {file_name}')