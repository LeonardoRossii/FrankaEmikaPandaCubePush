import os
import utils
from llm import GPT
from cem import CEM
from env import Push
import agent
import robosuite as suite
from openai import OpenAI
import importlib

utils.register_environment(Push, "Push")
controller = suite.load_controller_config(default_controller="OSC_POSE")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

env = suite.make(
    "Push",
    robots="Panda",
    controller_configs=controller,
    has_renderer=True,             
    has_offscreen_renderer=True,
    render_collision_mesh=False,
    use_camera_obs=False,
    render_camera="frontview",      
    control_freq=25,
    horizon = 250
)

llm = GPT(client)
llm.generate_preference_setup()
importlib.reload(agent)

obs = env.reset()
agent = agent.Agent(env)
opt = CEM(agent, llm, reward_gen=True)
opt.train()