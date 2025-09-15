import os
import numpy as np

import robosuite as suite
from robosuite.controllers import load_controller_config

from env import Push
from sac import Agent
import utils
from utils import plot_learning_curve


def flatten_obs(obs_dict):
    parts = []
    for k, v in obs_dict.items():
        if isinstance(v, np.ndarray):
            parts.append(v.ravel())
    return np.concatenate(parts, axis=0).astype(np.float32)


def scale_action(agent_action, low, high):
    agent_action = np.clip(agent_action, -1.0, 1.0)
    return low + 0.5 * (agent_action + 1.0) * (high - low)


if __name__ == "__main__":
    os.makedirs("tmp/sac", exist_ok=True) 
    utils.register_environment(Push, "Push")
    controller = load_controller_config(default_controller="OSC_POSE")
    table_full_size = (0.8, 0.8, 0.05)

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
        table_full_size=table_full_size,
        reward_shaping=True,
        horizon=1000,
        )

    act_low, act_high = env.action_spec 
    n_actions = act_low.shape[0]

    first_obs = env.reset()
    obs_dim = 9

    agent = Agent(
        input_dims=(obs_dim,),
        env=env,              
        n_actions=n_actions,
    )

    max_n_timesteps = 1000
    episodes = 1000
    filename = "robosuite_push_panda.png"
    figure_file = os.path.join("plots", filename)
    os.makedirs("plots", exist_ok=True)

    best_score = -np.inf
    score_history = []
    load_checkpoint = False

    if load_checkpoint:
        agent.load_models()

    for i in range(episodes):
        obs_dict = env.reset()
        
        state = np.concatenate([
            obs_dict["eef_to_cube"].ravel(),
            obs_dict["cube_to_goal"].ravel(),
            obs_dict["eef_to_goal"].ravel()
            ]).astype(np.float32)
        
        observation = state
        done = False
        score = 0.0
        gamma = 0.99

        for t in range(max_n_timesteps): 

            agent_action = agent.choose_action(observation)

            env_action = scale_action(agent_action, act_low, act_high)

            param = [1]
            obs_dict_, reward, done, info = env.step(env_action, param)
            reward = np.float32(reward[0])
            
            state = np.concatenate([
                obs_dict_["eef_to_cube"].ravel(),
                obs_dict_["cube_to_goal"].ravel(),
                obs_dict_["eef_to_goal"].ravel()
            ]).astype(np.float32)

            observation_ = state

            score += reward

            agent.remember(observation, agent_action, reward, observation_, done)

            if not load_checkpoint:
                agent.learn()

            observation = observation_

            if done or env.check_success() or env.check_failure():
                break

        score_history.append(score)
        avg_score = np.mean(score_history[-100:]) if len(score_history) > 0 else score

        if avg_score > best_score:
            best_score = avg_score
            if not load_checkpoint:
                agent.save_models()

        print(f"episode {i}  score {score:.1f}  avg_score {avg_score:.1f}")

    if not load_checkpoint:
        x = [i + 1 for i in range(episodes)]
        plot_learning_curve(x, score_history, figure_file)
