import numpy as np

def get_reward(env, action, params):
    rewards = []
    for param in params:
        reward = 0

        eef_to_cube_dist = np.linalg.norm(env.get_eef_pos() - env.get_cube_pos())
        cube_to_goal_dist = np.linalg.norm(env.get_cube_pos() - env.get_goal_pos())
        cube_to_boundary_dist = env.get_cube_bound_dist()

        reward += 1.0 - eef_to_cube_dist
        reward += 1.0 - cube_to_goal_dist

        if env.check_contact_finger_1_cube() and env.check_contact_finger_2_cube():
            reward += 0.5

        if env.check_contact_table():
            reward -= 0.01

        IE_reward = cube_to_boundary_dist

        max_rewards_bonuses = 2.5
        reward += 10 * env.horizon * max_rewards_bonuses * int(env.check_success())

        reward = param * reward + (1 - param) * IE_reward
        rewards.append(reward)
    return rewards

def get_success_condition(env):
    return np.linalg.norm(env.get_cube_pos() - env.get_goal_pos()) < 0.04

def get_failure_condition(env):
    return np.linalg.norm(env.get_eef_pos() - env.get_cube_pos()) > 0.2