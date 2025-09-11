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
agent = Agent(env, env.action_dim)
agent.set_weights(np.loadtxt("theta.txt"))

for step in range(250):
    print(env.check_contact_cube())
    state = agent.get_state(obs)
    action = agent.forward(state)
    obs, _, done, _ = env.step(action, [0])
    if done or env.check_success():
        break
    env.render()
env.close()