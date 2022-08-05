# imports
import json
import os
import shutil
import sys

# CHANGES HERE
# uncomment if error appears
# import os
# os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
# CHANGES END HERE
import ray
from ray.rllib.agents.dqn import DQNTrainer, DEFAULT_CONFIG

import wandb

sys.path.insert(0,"")
from Manhattan_Graph_Environment.gym_graphenv.envs.GraphworldManhattan import GraphEnv, CustomCallbacks

# login wandb
wandb.login(key="93aab2bcc48447dd2e8f74124d0258be2bf93859")
wandb.init(project="Comparison-Total_Env", entity="hitchhike")

env=GraphEnv()

# Initialize Ray
ray.init()

#Set trainer configuration 
trainer_config = DEFAULT_CONFIG.copy()
trainer_config['num_workers'] = 1
trainer_config["train_batch_size"] = 400
trainer_config["gamma"] = 0.95
trainer_config["n_step"] = 1
#trainer_config["framework"] = "torch"
trainer_config["callbacks"] = CustomCallbacks
#trainer_config["num_gpus"] = int(os.environ.get("RLLIB_NUM_GPUS", "0"))
trainer_config["explore"] = True
trainer_config["exploration_config"] = {
    "type": "EpsilonGreedy",
    "initial_epsilon": 1.0,
    "final_epsilon": 0.02,
}

# Initialize trainer
trainer = DQNTrainer(trainer_config,GraphEnv )

#Define the path where the results of the trainer should be saved
checkpoint_root = "tmp/dqn/graphworld"
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
                'share_delivered_on_time': float(result["count_delivered_on_time"]/result['episodes_this_iter']),
               'boolean_has_booked_any_own': int(result["boolean_has_booked_any_own"]),
               'count_shared_available': int(result["count_shared_available"]),
               "ratio_shared_available_to_all_steps": float(result["ratio_shared_available_to_all_steps"]),
               'count_shared_available_useful': int(result["count_shared_available_useful"]),
               'shared_taken_to_shared_available': float(result["shared_taken_to_shared_available"]),
               'shared_available_useful_to_shared_available': float(result["shared_available_useful_to_shared_available"]),
               'shared_taken_useful_to_shared_available_useful': float(result["shared_taken_useful_to_shared_available_useful"]),
               'ratio_delivered_without_bookown_to_all_delivered': float(result["ratio_delivered_without_bookown_to_all_delivered"]),
                'bookown_distance_not_covered_share':float(result['bookown_distance_not_covered_share_mean']),
                'bookown_distance_not_covered': float(result['bookown_distance_not_covered_mean']),
                'distance_reduced_with_ownrides':float(result['distance_reduced_with_ownrides_mean']),
                'distance_reduced_with_shared':float(result['distance_reduced_with_shared_mean']),
                'distance_reduced_with_ownrides_share':float(result['distance_reduced_with_ownrides_share_mean']),
                'distance_reduced_with_shared_share':float(result['distance_reduced_with_shared_share_mean']),
               }
    episode_data.append(episode)
    episode_json.append(json.dumps(episode))
    file_name = trainer.save(checkpoint_root)
    wandb.save(file_name)
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
                'share_delivered_on_time': result["count_delivered_on_time"]/result['episodes_this_iter'],
                'boolean_has_booked_any_own': result["boolean_has_booked_any_own"],
                'count_shared_available': result["count_shared_available"],
                'count_shared_available_useful': result["count_shared_available_useful"],
                'shared_taken_to_shared_available': result["shared_taken_to_shared_available"],
                'shared_available_useful_to_shared_available': result["shared_available_useful_to_shared_available"],
                'shared_taken_useful_to_shared_available_useful': result["shared_taken_useful_to_shared_available_useful"],
                "ratio_shared_available_to_all_steps": result["ratio_shared_available_to_all_steps"],
                "ratio_delivered_without_bookown_to_all_delivered": result["ratio_delivered_without_bookown_to_all_delivered"],
                'bookown_distance_not_covered_share': result['bookown_distance_not_covered_share_mean'],
                'bookown_distance_not_covered': result['bookown_distance_not_covered_mean'],
                'distance_reduced_with_ownrides':result['distance_reduced_with_ownrides_mean'],
                'distance_reduced_with_shared':result['distance_reduced_with_shared_mean'],
                'distance_reduced_with_ownrides_share':result['distance_reduced_with_ownrides_share_mean'],
                'distance_reduced_with_shared_share':result['distance_reduced_with_shared_share_mean'],

    })

    print(f'{n + 1:3d}: Min/Mean/Max reward: {result["episode_reward_min"]:8.4f}/{result["episode_reward_mean"]:8.4f}/{result["episode_reward_max"]:8.4f}, len mean: {result["episode_len_mean"]:8.4f}. Checkpoint saved to {file_name}')