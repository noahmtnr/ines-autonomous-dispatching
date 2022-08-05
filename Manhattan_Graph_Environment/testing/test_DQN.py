"""
Test Class for DQN Agent.
"""

# imports
import sys

import gym
import ray
from ray.rllib.agents.dqn import DQNTrainer, DEFAULT_CONFIG

sys.path.insert(0,"")

from gym_graphenv.envs.GraphworldManhattan import GraphEnv

env=GraphEnv()

file_name = "results/tmp/dqn/graphworld"

trainer_config = DEFAULT_CONFIG.copy()
trainer_config['num_workers'] = 1
trainer_config["train_batch_size"] = 400
trainer_config["gamma"] = 0.95
trainer_config["n_step"] = 10
trainer_config["framework"] = "torch"
#num_gpus and other gpu parameters in order to train with gpu

trainer = DQNTrainer(trainer_config,GraphEnv )
trainer.restore(file_name)
env = gym.make("graphworld-v0")
state = env.reset()

sum_reward = 0
n_step = 20
for step in range(n_step):
    action = trainer.compute_action(state)
    state, reward, done, info = env.step(action)
    sum_reward += reward
    #env.render()
    if done == 1:
        print("cumulative reward", sum_reward)
        state = env.reset()
        sum_reward = 0

ray.shutdown()