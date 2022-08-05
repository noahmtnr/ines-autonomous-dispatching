from gym.envs.registration import register

register(
    id="graphworld-v0",
    entry_point="gym_graphenv.envs:GraphEnv",
)
