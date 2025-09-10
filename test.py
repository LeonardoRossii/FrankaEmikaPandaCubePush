import numpy as np
import utils
from agent import Agent
import robosuite as suite
from env import Push

utils.register_environment(Push, "Push")
controller = suite.load_controller_config(default_controller="OSC_POSE")

env = suite.make(
    "Push",
    robots="Panda",
    controller_configs=controller,
    has_renderer=True,             
    has_offscreen_renderer=False,
    use_camera_obs=False,
    render_camera="sideview",      
    control_freq=25,             
)

obs = env.reset()

input_dim = 2
output_dim = 7
agent = Agent(env, input_dim, output_dim)
agent.set_weights(np.loadtxt("theta.txt"))

for step in range(250):
    state = agent.get_state(obs)
    action = agent.forward(state)
    obs, reward, done, info = env.step(action)
    if done or env.check_success():
        obs = env.reset()
        break
    env.render()
env.close()