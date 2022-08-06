import gym


# from ray.tune.registry import register_env
# import ray
# import ray.rllib.agents.ppo as ppo
def run_one_episode(env):
    env.reset()
    sum_reward = 0
    for i in range(env.MAX_STEPS):
        action = env.action_space.sample()
        state, reward, done, info = env.step(action)
        sum_reward += reward
        # print(sum_reward)
        # env.render()
        if done:
            break
    return sum_reward


env = gym.make("gridworld-v0")
sum_reward = run_one_episode(env)

history = []
for _ in range(100):
    sum_reward = run_one_episode(env)
    history.append(sum_reward)
avg_sum_reward = sum(history) / len(history)
print("\nbaseline cumulative reward: {:6.2}".format(avg_sum_reward))

# def main():
#     ray.init(ignore_reinit_error=True)

#     # register the custom environment
#     select_env = "gridworld-v0"
#     #select_env = "fail-v1"
#     register_env(select_env, lambda config: Gridworld_v0())
#     #register_env(select_env, lambda config: Fail_v1())


#     # configure the environment and create agent
#     config = ppo.DEFAULT_CONFIG.copy()
#     config["log_level"] = "WARN"
#     agent = ppo.PPOTrainer(config, env=select_env)

#     status = "{:2d} reward {:6.2f}/{:6.2f}/{:6.2f} len {:4.2f} saved {}"
#     n_iter = 5

#     # train a policy with RLlib using PPO
#     for n in range(n_iter):
#         result = agent.train()
#         #chkpt_file = agent.save(chkpt_root)

#         print(status.format(
#                 n + 1,
#                 result["episode_reward_min"],
#                 result["episode_reward_mean"],
#                 result["episode_reward_max"],
#                 result["episode_len_mean"],
#                 #chkpt_file
#                 ))


#     # examine the trained policy
#     policy = agent.get_policy()
#     model = policy.model
#     print(model.base_model.summary())
# main()
