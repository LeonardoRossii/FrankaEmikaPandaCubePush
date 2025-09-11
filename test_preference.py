import numpy as np
import utils
import llm
from agent import Agent
import robosuite as suite
from env import Push
from pathlib import Path

utils.register_environment(Push, "Push")
controller = suite.load_controller_config(default_controller="OSC_POSE")

env = suite.make(
    "Push",
    robots="Panda",
    controller_configs=controller,
    has_renderer=True,             
    has_offscreen_renderer=False,
    use_camera_obs=False,
    render_camera="sideview",      
    control_freq=25,     
)

obs = env.reset()
agent = Agent(env, env.action_dim)

weight_1 = np.loadtxt("theta.txt")
weight_2 = np.loadtxt("Theta.txt")
weights = [weight_1, weight_2]


current_dir = Path(__file__).parent

file_task_description_path = current_dir / "prompt_.txt"
with open(file_task_description_path, "r") as file:
    prompt = file.read().strip()

content = llm.get_preference(agent, weights, 250, prompt)
print(content)
env.close()
