import numpy as np

def get_reward(env, action, params):

    eef_pos = env.get_eef_pos()
    cube_pos = env.get_cube_pos()
    goal_pos = env.get_goal_pos()

    eef_to_cube_dist = np.linalg.norm(eef_pos - cube_pos)
    cube_to_goal_dist = np.linalg.norm(cube_pos - goal_pos)

    rewards = []

    for param in params:

        reward = 0.0

        reward += 1.0 - eef_to_cube_dist
        reward += 1.0 - cube_to_goal_dist

        safe_reward = env.get_cube_bound_dist()

        if env.check_contact_cube():
            reward += 0.5

        if env.check_contact_table():
            reward -= 0.1

        if cube_to_goal_dist < 0.04:
            terminal_reward_term = 10 * env.horizon * (2.0 + 0.5)
            reward += terminal_reward_term

        reward = param*reward + (1-param)*safe_reward

        rewards.append(reward)
    return rewards

def get_success_condition(env):
    cube_pos = env.get_cube_pos()
    goal_pos = env.get_goal_pos()
    return np.linalg.norm(cube_pos - goal_pos) < 0.04

def get_failure_condition(env):
    eef_pos = env.get_eef_pos()
    cube_pos = env.get_cube_pos()
    return np.linalg.norm(eef_pos - cube_pos) > 0.2