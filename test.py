import numpy as np
import utils
from agent import Agent
from agent import NNAgent
from filters import Filter
from filters import FilterCBF
import robosuite as suite
from env import Push

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
    has_renderer=True,             
    has_offscreen_renderer=False,
    use_camera_obs=False,
    render_camera="sideview",      
    control_freq=25,   
    table_full_size = table_full_size,  
)

obs = env.reset()

agent = NNAgent(env, env.action_dim)
sfae_filter = Filter(env)
agent.set_weights(np.loadtxt("theta.txt"))

for step in range(250):
    state = agent.get_state(obs)
    action = agent.forward(state)
    action = sfae_filter.apply(action)
    obs, _, done, _ = env.step(action, [0])
    print(env.check_success())
    if done or env.check_success() or env.check_failure():
        break
    env.render()
env.close()