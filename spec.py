import numpy as np

def get_reward(env, action, params):
    rewards = []
    for param in params:
        reward = 0

        eef_pos = env.get_eef_pos()
        cube_pos = env.get_cube_pos()
        goal_pos = env.get_goal_pos()
        
        eef_to_cube_dist = np.linalg.norm(eef_pos - cube_pos)
        cube_to_goal_dist = np.linalg.norm(cube_pos - goal_pos)

        reward += 1.0 - eef_to_cube_dist
        reward += 1.0 - cube_to_goal_dist

        if env.check_contact_cube():
            reward += 1        

        if env.check_contact_table():
            reward -= 0.05

        IE_reward = 0
        reward = param * reward + (1 - param) * IE_reward

        if env.check_success():
            max_rewards_bonuses = 2.5
            reward += max_rewards_bonuses*10

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