import os
import json
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
controller = suite.load_controller_config(default_controller="OSC_POSE")
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
    horizon = 250
)

_ = env_.reset()
agent = Agent(env_)
safe_filter = FilterCBF(env_)
llm = GPT(client)

weights = np.loadtxt(Path("weights") / "weights.txt")
null = np.zeros(len(weights))

print(llm.generate_irreversible_events())
llm.generate_preference_setup()

ms = []

_, m0 = agent.evaluate(weights,
                      env_.horizon,
                      render=False,
                      trajectory=True)
ms.append(m0)

_, m1 = agent.evaluate(null,
                       env_.horizon,
                       render=False,
                       trajectory=True)

ms.append(m1)

tdesc = "\n\n".join(f"Trajectory {i+1}:\n{json.dumps(m, indent=3)}" for i, m in enumerate(ms))

llm.generate_preference(tdesc)

env_.close()

