import os 
os.environ['HDF5_DISABLE_VERSION_CHECK']='2'
from gym_graphenv.envs.GraphworldManhattan import GraphEnv
import numpy as np
def run_one_episode (env):
    env.reset()
    print("reset done")
    sum_reward = 0
    counter = 0
    while(counter<30):
        # visualize current situation
        env.render()

        # look in adjacency matrix for costs from the current position
        array = env.learn_graph.adjacency_matrix('cost')[env.position].astype(int)
        min = np.amin(array)
        array = np.where(array==min,50,array)
        
        # get minimal value in array
        #while(action==env.position):
        min_value = np.amin(array)
        print(min_value)
        # if multiple entries have the same value
        all_indices = np.where(array==min_value)
        print(f"Alle: {all_indices[0]}")
        action = np.random.choice(all_indices[0])
        print(action)
        # select random of all_indices
            #while(action==env.position):
            #    action = np.random.choice(all_indices[0])

        # select action and show it
        #action = env.action_space[dest_hub]
        print(f"Our destination hub is: {action}")
        state, reward, done, info = env.step(action)

        # add reward
        sum_reward+=reward
        
        if done:
            print("DELIVERY DONE! sum_reward: ",sum_reward)
            break

        print("sum_reward: ",sum_reward)
        # print("sum_reward: ",sum_reward, " time: ",env.time, "deadline time: ", env.deadline, "pickup time: ", env.pickup_time)
    return sum_reward

def main():
    env=GraphEnv()
    run_one_episode(env)

main()
