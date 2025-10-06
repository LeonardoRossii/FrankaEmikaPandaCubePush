import os
import utils
from llm import GPT
from cem import CEM
from env import Push
from agent import Agent
import robosuite as suite
from openai import OpenAI

utils.register_environment(Push, "Push")
controller = suite.load_controller_config(default_controller="OSC_POSE")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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
    horizon = 250      
)

obs = env.reset()
agent = Agent(env)
llm = GPT(client)
opt = CEM(agent, llm, reward_gen=False)
opt.train()