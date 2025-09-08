import math
import numpy as np
import robosuite as suite
from robosuite.environments import ALL_ENVIRONMENTS
from env import Push
from agent import Agent

if "Push" not in ALL_ENVIRONMENTS: ALL_ENVIRONMENTS["Push"] = Push
controller = suite.load_controller_config(default_controller="OSC_POSE")

env = suite.make(
    "Push",
    robots="Panda",
    controller_configs=controller,
    has_renderer=True,             
    has_offscreen_renderer=False,
    use_camera_obs=False,
    render_camera="sideview",      
    control_freq=20,             
)

obs = env.reset()

input_dim = 6
output_dim = 7
agent = Agent(env, input_dim, output_dim)
agent.set_weights(np.loadtxt("policy.txt"))

for step in range(250):
    state = agent.get_state(obs)
    action = agent.forward(state)
    obs, reward, done, info = env.step(action)
    print("Contact? : ", env.check_contact(env.robots[0].gripper, env.cube))
    print("Distance to goal: ", env.dist_to_goal())
    print("Contact Table: ", env.check_contact_table())
    print("Check Success: ", env._check_success())
    if done or env._check_success():
        obs = env.reset()
        break
    env.render()
env.close()
