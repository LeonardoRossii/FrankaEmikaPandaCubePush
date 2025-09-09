import numpy as np
import robosuite as suite
from env import Push 
from agent import Agent
from cem import cem
import reward
from importlib import reload
from rewrite_reward import rewrite_reward 
from robosuite.environments import ALL_ENVIRONMENTS

if "Push" not in ALL_ENVIRONMENTS: ALL_ENVIRONMENTS["Push"] = Push
controller = suite.load_controller_config(default_controller="OSC_POSE")

hasRewrite = False
if hasRewrite:
    file_path = f"/home/leojellypc/cube_push/reward_prompt.txt"
    with open(file_path, "r") as file:
        prompt = file.read().strip()
    rewrite_reward(prompt)
    reload(reward)

env = suite.make(
    "Push",
    robots="Panda",
    controller_configs=controller,
    has_renderer=False,             
    has_offscreen_renderer=False,
    render_collision_mesh=False,
    use_camera_obs=False,
    render_camera=None,      
    control_freq=25,          
)

env.sim.model.opt.timestep = 0.01
env.sim.nsubsteps = 1               

obs = env.reset()
agent = Agent(env,2,7)
_ = cem(agent, max_n_timesteps=250)