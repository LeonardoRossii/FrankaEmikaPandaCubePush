import os
import numpy as np
import utils
from agent import Agent
from filters import FilterCBF
import robosuite as suite
from env import Push
from pathlib import Path
from llm import GPT
from openai import OpenAI

utils.register_environment(Push, "Push")
controller = suite.load_controller_config(default_controller="JOINT_VELOCITY")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

env_ = suite.make(
    "Push",
    robots="Panda",
    controller_configs=controller,
    has_renderer=True,             
    has_offscreen_renderer=True,
    use_camera_obs=True,
    render_camera="sideview",      
    control_freq=25,
    horizon = 500
)

_ = env_.reset()
agent = Agent(env_)
safe_filter = FilterCBF(env_)
llm = GPT(client)
weights = np.loadtxt(Path("weights") / "weights.txt")
_,_,_ = agent.evaluate(weights, env_.horizon, [0], render=True)
env_.close()

