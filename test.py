import env
import numpy as np
import utils
from agent import Agent
from filters import FilterCBF
import robosuite as suite
from env import Push
from pathlib import Path

print(suite.__version__)
utils.register_environment(Push, "Push")
controller = suite.load_controller_config(default_controller="OSC_POSE")

print(utils.extract_function_from_class(env.Push, '_setup_observables'))

env_ = suite.make(
    "Push",
    robots="Panda",
    controller_configs=controller,
    has_renderer=True,             
    has_offscreen_renderer=True,
    use_camera_obs=True,
    render_camera="sideview",      
    control_freq=25,
    horizon = 250
)

_ = env_.reset()
agent = Agent(env_)
safe_filter = FilterCBF(env_)
weights = np.loadtxt(Path("weights") / "weights.txt")
agent.evaluate(weights, env_.horizon, render=True)