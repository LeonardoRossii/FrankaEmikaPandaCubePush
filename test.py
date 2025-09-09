import numpy as np
from env import Push
from agent import Agent
import robosuite as suite
from robosuite.environments import ALL_ENVIRONMENTS

if "Push" not in ALL_ENVIRONMENTS: ALL_ENVIRONMENTS["Push"] = Push
controller = suite.load_controller_config(default_controller="OSC_POSE")
policy_params = "policy_.txt"

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
agent.set_weights(np.loadtxt(policy_params))

for step in range(250):
    state = agent.get_state(obs)
    action = agent.forward(state)
    obs, reward, done, info = env.step(action)
    if done or env._check_success():
        obs = env.reset()
        break
    env.render()
env.close()