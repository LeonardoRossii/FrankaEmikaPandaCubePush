import cem
import llm
import spec
import utils
import importlib
from env import Push
from agent import Agent
from pathlib import Path
import robosuite as suite

generate_new_task_spec = False
if generate_new_task_spec:
    
    current_dir = Path(__file__).parent

    file_task_description_path = current_dir / "pmptspec.txt"
    with open(file_task_description_path, "r") as file:
        prompt = file.read().strip()

    file_environment_class_path = current_dir / "env.py"
    with open(file_environment_class_path, "r") as file:
        env_class = file.read().strip()

    prompt += "\n\n# Environment class: \n" + env_class

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

obs = env.reset()
agent = Agent(env, env.action_dim)
cem.cem(agent, max_n_timesteps=250)