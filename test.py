import numpy as np
import utils
from agent import Agent
from filters import Filter
from filters import FilterCBF
import robosuite as suite
from env import Push

utils.register_environment(Push, "Push")
controller = suite.load_controller_config(default_controller="OSC_POSE")

table_x_width = 0.8
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
    render_camera="frontview",      
    control_freq=25,   
    table_full_size = table_full_size,
    horizon = 1000
)

obs = env.reset()

agent = Agent(env, env.action_dim)
safe_filter = FilterCBF(env)
agent.set_weights(np.loadtxt("theta_drop_1.txt"))
nom_action = np.array([0.0,0.1,0,0,0,0,0])

for step in range(1000):
    state = agent.get_state(obs)
    action = agent.forward(state)
    safe_action = safe_filter.apply(action)
    obs, _, done, _ = env.step(safe_action, [0])
    if done or env.check_success() or env.check_failure():
        break
    env.render()
env.close()