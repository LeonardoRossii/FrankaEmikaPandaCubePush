import numpy as np
import robosuite as suite
from env import Push 
from agent import Agent
from cem import cem
from robosuite.environments import ALL_ENVIRONMENTS

if "Push" not in ALL_ENVIRONMENTS: ALL_ENVIRONMENTS["Push"] = Push
controller = suite.load_controller_config(default_controller="OSC_POSE")

env = suite.make(
    "Push",
    robots="Panda",
    controller_configs=controller,
    has_renderer=False,             
    has_offscreen_renderer=False,
    render_collision_mesh=False,
    use_camera_obs=False,
    render_camera=None,      
    control_freq=100,          
)

env.sim.model.opt.timestep = 0.01
env.sim.nsubsteps = 1               

obs = env.reset()
agent = Agent(env,6,7)
_ = cem(agent, max_n_timesteps=250)