import numpy as np
import pandas as pd
import json
import os
import shutil
import sys
import gym
import wandb

import ray
from ray.rllib.agents.dqn import DQNTrainer, DEFAULT_CONFIG

sys.path.insert(0,"")

from Manhattan_Graph_Environment.graphs.ManhattanGraph import ManhattanGraph
from gym_graphenv.envs.GraphworldManhattan import GraphEnv

wandb.login(key="93aab2bcc48447dd2e8f74124d0258be2bf93859")
wandb.init(project="Manhattan-env-test10", entity="hitchhike")

env=GraphEnv()


# Initialize Ray
ray.init()

#Set trainer configuration
trainer_config = DEFAULT_CONFIG.copy()
trainer_config['num_workers'] = 1
trainer_config["train_batch_size"] = 400
trainer_config["gamma"] = 0.95
trainer_config["n_step"] = 10
trainer_config["framework"] = "torch"
#trainer_config["num_gpus"] = int(os.environ.get("RLLIB_NUM_GPUS", "0"))


# Initialize trainer
trainer = DQNTrainer(trainer_config,GraphEnv )

#Define the path where the results of the trainer should be saved
checkpoint_root = "results/tmp/dqn/graphworld"
shutil.rmtree(checkpoint_root, ignore_errors=True, onerror=None)   # clean up old runs


# Run trainer

results = []
episode_data = []
episode_json = []
n_iter = 10

for n in range(n_iter):
    result = trainer.train()
    results.append(result)
    print("TEST", n)

    episode = {'n': n,
               'episode_reward_min': result['episode_reward_min'],
               'episode_reward_mean': result['episode_reward_mean'],
               'episode_reward_max': result['episode_reward_max'],
               'episode_len_mean': result['episode_len_mean'],
               }

    episode_data.append(episode)
    episode_json.append(json.dumps(episode))
    file_name = trainer.save(checkpoint_root)
    wandb.log({"mean_reward": result['episode_reward_mean']})

    print(f'{n + 1:3d}: Min/Mean/Max reward: {result["episode_reward_min"]:8.4f}/{result["episode_reward_mean"]:8.4f}/{result["episode_reward_max"]:8.4f}, len mean: {result["episode_len_mean"]:8.4f}. Checkpoint saved to {file_name}')