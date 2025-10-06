import numpy as np

def get_reward(env, _):
    lambda_ = 0.5
    reward = 0

    eef_pos = env.get_eef_pos()
    cube_pos = env.get_cube_pos()
    goal_pos = env.get_goal_pos()

    eef_to_cube_dist = np.linalg.norm(eef_pos - cube_pos)
    cube_to_goal_dist = np.linalg.norm(cube_pos - goal_pos)
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

    reward = lambda_ * reward + (1 - lambda_) * IE_reward

    return reward

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