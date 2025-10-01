import cem
import llm
import spec
import utils
import importlib
from env import Push
from agent import Agent
from pathlib import Path
import robosuite as suite
import matplotlib.pyplot as plt

generate_new_task_spec = True
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

table_x_width = 0.80
table_y_width = 0.15
table_z_width = 0.05
table_full_size = (table_x_width,
                   table_y_width,
                   table_z_width)

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
    table_full_size= table_full_size         
)

obs = env.reset()
agent = Agent(env, env.action_dim)
drops, drops_vec, lambda_vec = cem.cem(agent, max_n_timesteps=250)


print("Total drops:", drops)

plt.plot(drops_vec, marker='o', linestyle='-', color='b')
plt.title("Cumulative drops")
plt.xlabel("Episodes")
plt.ylabel("Value")
plt.grid(True)
plt.show()

plt.plot(lambda_vec, marker='o', linestyle='-', color='b')
plt.title("Lambda")
plt.xlabel("CEM iteration")
plt.ylabel("Value")
plt.grid(True)
plt.show()