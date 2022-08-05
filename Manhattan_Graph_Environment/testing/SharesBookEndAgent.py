"""
SharesBookEnd Agent.
Takes the shared ride that reduces the remaining distance the most in each step.
If no shared ride is available, the agent waits at the current hub.
If the agent hasn't reached the final hub two hours before deadline, he is forced to book an own ride to the final hub.
"""

import sys
sys.path.insert(0,"")
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class SharesBookEndAgent:

    """
    Runs the agent in the environment (by taking steps according to policy of agent) until it reaches the final hub.
    :param env: 
    :param reward_list:
    :param env_config:
    :return: dictionary containing results of run.
    """
    def run_one_episode (env,reward_list,env_config):
        env.reset()
        print("reset done")
        counter = 0
        sum_reward = 0
        sum_travel_time = timedelta(seconds=0)
        sum_distance = 0
        route = [env_config["pickup_hub_index"]]
        route_timestamps = [datetime.strptime(env_config["pickup_timestamp"], '%Y-%m-%d %H:%M:%S')]
        done = False
        current_hub = env_config["pickup_hub_index"]
        time_until_deadline = timedelta(hours=24)
        count_shares = 0
        count_bookowns = 0
        count_wait = 0
        steps = 0

        # run until finished or two hours before deadline are reached
        while (not done) and (time_until_deadline.total_seconds()/60 >= 120):
            # visualize current situation
            # env.render()

            # select most useful shared ride, otherwise wait
            best_gain = 0
            best_hub = 0
            print("distinction: ", env.state["distinction"])
            for hub in range(env.action_space.n):
                # check distance gained
                if (env.state["remaining_distance"][hub] > 0) and (env.state["remaining_distance"][hub] > best_gain) and (env.state["distinction"][hub] == 1):
                    best_hub = hub
                    best_gain = env.state["remaining_distance"][hub]

            if best_gain <= 0:
                action = current_hub
            else:
                action = best_hub

            print(f"Our destination hub is: {action}")
            state, reward, done, info = env.step(action)

            # get information from action
            route.append(action)
            current_hub = action
            route_timestamps.append(info.get('timestamp'))
            sum_travel_time +=timedelta(seconds=info.get('step_travel_time'))
            delivey_time = datetime.strptime(env_config["delivery_timestamp"], '%Y-%m-%d %H:%M:%S')
            time_until_deadline= timedelta(hours=24)-sum_travel_time
            sum_distance += info.get('distance')/1000
            number_hubs=info.get('count_hubs')
            dist_shares = info.get("dist_covered_shares")
            dist_bookowns = info.get("dist_covered_bookown")
            # add reward
            sum_reward+=reward
            action_choice = info.get("action")

            if action_choice == "Share":
                count_shares += 1
            elif action_choice == "Book":
                count_bookowns += 1
            elif action_choice == "Wait":
                count_wait += 1
            steps += 1
            
            if done:
                print("DELIVERY DONE! sum_reward: ",sum_reward)
                print("DELIVERY DONE! Route: ",route)
                print("DELIVERY DONE! Travel Time: ",sum_travel_time)
                print("DELIVERY DONE! Distance: ",sum_distance)
                print("DELIVERY DONE! Hubs: ",number_hubs)
                print("DELIVERY DONE! unitl deadline: ",time_until_deadline)
                # if action!=env_config["delivery_hub_index"]:
                #     raise Exception("DID NOT ARRIVE IN FINAL HUB")
                break

        # agent needs to book own ride if deadline is closer than 2 hours
        if(time_until_deadline.total_seconds()/60 < 120):
            print("Force Manual Delivery")
            action = env_config["delivery_hub_index"]
            # action = final hub
            state, reward, done, info = env.step(action)
            done = True
            route.append(action)
            print("Timestamps",info.get('timestamp') )
            route_timestamps.append(info.get('timestamp'))
            sum_reward += reward
            sum_travel_time +=timedelta(seconds=info.get('step_travel_time'))
            delivey_time = datetime.strptime(env_config["delivery_timestamp"], '%Y-%m-%d %H:%M:%S')
            time_until_deadline= timedelta(hours=24) - sum_travel_time
            sum_distance += info.get('distance')/1000
            number_hubs=info.get('count_hubs')
            action_choice = info.get("action")
            dist_shares = info.get("dist_covered_shares")
            dist_bookowns = info.get("dist_covered_bookown")

            if action_choice == "Share":
                count_shares += 1
            elif action_choice == "Book":
                count_bookowns += 1
            elif action_choice == "Wait":
                count_wait += 1
            steps += 1

        if count_bookowns == 0:
            ratio = 0
        else:
            ratio = float(count_shares/count_bookowns)
        
        # results of the agent's run
        reward_list={"pickup_hub":env_config['pickup_hub_index'],"delivery_hub":env_config['delivery_hub_index'],"reward":sum_reward, "hubs":number_hubs, "route":route, "time":sum_travel_time, "dist":sum_distance, "time_until_deadline":time_until_deadline, "timestamps":route_timestamps, "count_bookowns": count_bookowns, "steps": steps, "ratio_share_to_own": ratio,"dist_covered_shares": dist_shares, "dist_covered_bookown": dist_bookowns}
        return reward_list
