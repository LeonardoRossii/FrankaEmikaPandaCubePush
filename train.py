import cem
import llm
import spec
import utils
import importlib
from env import Push
from agent import Agent
import robosuite as suite
from pathlib import Path

current_dir = Path(__file__).parent
file_path = current_dir / "prompt.txt"
with open(file_path, "r") as file:
    prompt = file.read().strip()
llm.generate_spec(prompt)
importlib.reload(spec)

utils.register_environment(Push, "Push")
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
    control_freq=25,          
)
env.sim.nsubsteps = 1
env.sim.model.opt.timestep = 0.01
       
obs = env.reset()
agent = Agent(env, 2, 7)
_ = cem.cem(agent, max_n_timesteps=250)