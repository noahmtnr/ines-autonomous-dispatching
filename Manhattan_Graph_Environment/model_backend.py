import gym
import ray
import ray.rllib.agents.dqn as dqn
from ray import serve
from starlette.requests import Request
from ray.rllib.agents.dqn import DQNTrainer, DEFAULT_CONFIG
from gym_graphenv.envs.GraphworldManhattan import GraphEnv
import gym
import requests
# checkpoint_path = "C:\\Users\\cosmi\\Documents\\Mannheim\\ines-autonomous-dispatching\\results\\tmp\\dqn\\graphworld\\checkpoint_000001\\checkpoint-1"

checkpoint_path = '/Users/noah/Desktop/Repositories/ines-autonomous-dispatching/Manhattan_Graph_Environment/training/results/tmp/dqn/graphworld/checkpoint_000001/checkpoint-1'
rainbow_config = DEFAULT_CONFIG.copy()
rainbow_config['num_workers'] = 3
rainbow_config["train_batch_size"] = 400
rainbow_config["gamma"] = 0.99
# rainbow_config["framework"] = "torch"
# rainbow_config["callbacks"] = CustomCallbacks
rainbow_config["n_step"]= 3 #[between 1 and 10]  //was 5 and 7
rainbow_config["noisy"] = True
rainbow_config["num_atoms"] = 70 #[more than 1] //was 51,20
rainbow_config["v_min"] =-15000
rainbow_config["v_max"]=10000 # (set v_min and v_max according to your expected range of returns).


@serve.deployment(route_prefix="/graphworld-rainbow")
class ServeRainbowModel:
    def __init__(self, checkpoint_path) -> None:
        self.trainer = dqn.DQNTrainer(
            config=rainbow_config,
            env=GraphEnv,
        )
        self.trainer.restore(checkpoint_path)

    async def __call__(self, request: Request):
        json_input = await request.json()
        obs = json_input["observation"]

        action = self.trainer.compute_single_action(obs)
        
        return {"action": int(action)}

# serve.start()
# ServeRainbowModel.deploy(checkpoint_path)

# list_hubs=[]
list_nodes=[]
env = GraphEnv()
obs = env.reset()
print(obs)
list_hubs=[env.position]
list_actions=["start"]
action_list=[0,7,1,1,13]
for i in action_list:
   

    # print(f"-> Sending observation {obs}")
    # resp = requests.get(
    #     "http://localhost:8000/graphworld-rainbow", json={"observation": obs.tolist()}
    # )
   
    # print(f"<- Received response {resp.json()}")
    resp={'action': i}
    action = resp["action"]
    print(action)
    obs, reward, done, info = env.step(action)
    if(info["route"][-1] != env.final_hub and info["action"] != "Wait"):
        print(list_nodes)
        list_nodes.extend(info["route"][0:-1])
        print(list_nodes)
    list_hubs.append(action)
    list_actions.append(info["action"])
    

    print("infos",info)
    print("hubs",list_hubs)
    print("nodes",list_nodes)
    print("actions",list_actions)