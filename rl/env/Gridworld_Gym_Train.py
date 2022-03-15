from doctest import Example
import gym
import gym_example
def run_one_episode (env):
    env.reset()
    sum_reward = 0
    for i in range(env.MAX_STEPS):
        action = env.action_space.sample()
        state, reward, done, info = env.step(action)
        print()
        env.render()
        if done:
            break
    return sum_reward

env = gym.make("gridworld-v0")
sum_reward = run_one_episode(env)

history = []
for _ in range(10):
    sum_reward = run_one_episode(env)
    history.append(sum_reward)
avg_sum_reward = sum(history) / len(history)
print("\nbaseline cumulative reward: {:6.2}".format(avg_sum_reward))