import ray
import ray.rllib.agents.dqn as dqn
from ray import serve
from starlette.requests import Request
from ray.rllib.agents.dqn import DQNTrainer, DEFAULT_CONFIG

checkpoint_path = '/Users/noah/Desktop/Repositories/ines-autonomous-dispatching/Manhattan_Graph_Environment/tmp/dqn/graphworld/checkpoint_000010/checkpoint-10'
rainbow_config = DEFAULT_CONFIG.copy()
rainbow_config['num_workers'] = 3
rainbow_config["train_batch_size"] = 400
rainbow_config["gamma"] = 0.99
rainbow_config["framework"] = "torch"
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
            env="graphworld-v0",
        )
        self.trainer.restore(checkpoint_path)

    async def __call__(self, request: Request):
        json_input = await request.json()
        obs = json_input["observation"]

        action = self.trainer.compute_single_action(obs)
        return {"action": int(action)}

serve.start()
ServeRainbowModel.deploy(checkpoint_path)