import numpy as np
import utils
from agent import Agent
from filters import FilterCBF
import robosuite as suite
from env import Push

utils.register_environment(Push, "Push")
controller = suite.load_controller_config(default_controller="OSC_POSE")

env = suite.make(
    "Push",
    robots="Panda",
    controller_configs=controller,
    has_renderer=True,             
    has_offscreen_renderer=False,
    use_camera_obs=False,
    render_camera="frontview",      
    control_freq=25,
    horizon = 250
)

_ = env.reset()
agent = Agent(env)
safe_filter = FilterCBF(env)
weights = np.loadtxt("weights.txt")
agent.evaluate(weights, env.horizon, render=True)