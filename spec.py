import numpy as np

def get_reward(env, action, params):
    rewards = []
    for param in params:
        reward = 0

        eef_pos = env.get_eef_pos()
        cube_pos = env.get_cube_pos()
        goal_pos = env.get_goal_pos()

        eef_cube_dist = np.linalg.norm(eef_pos - cube_pos)
        cube_goal_dist = np.linalg.norm(cube_pos - goal_pos)
        cube_bound_dist = env.get_cube_bound_dist()

        reward += 1.0 / (eef_cube_dist + 1e-3)
        reward += 1.0 / (cube_goal_dist + 1e-3)
        reward += 0.1 * cube_bound_dist

        if env.check_contact_finger_1_cube() or env.check_contact_finger_2_cube():
            reward += 0.5

        if env.check_contact_table():
            reward -= 0.01

        max_rewards_bonuses = 1.0 / 0.04 + 1.0 / 0.04 + 0.1 * env.table_full_size[1] / 2
        reward += 10 * env.horizon * max_rewards_bonuses * int(env.check_success())

        IE_reward = 0.1 * cube_bound_dist
        reward = param * reward + (1 - param) * IE_reward

        rewards.append(reward)
    return rewards

def get_success_condition(env):
    cube_pos = env.get_cube_pos()
    goal_pos = env.get_goal_pos()
    success = np.linalg.norm(cube_pos - goal_pos) < 0.04
    return success

def get_failure_condition(env):
    eef_pos = env.get_eef_pos()
    cube_pos = env.get_cube_pos()
    failure = np.linalg.norm(eef_pos - cube_pos) > 0.2
    return failure