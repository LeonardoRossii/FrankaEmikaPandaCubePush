import numpy as np

def get_reward(env, action):
    max_dist = 0.2
    T = 250
    terminal_reward = 1000

    eef_pos = env.sim.data.site_xpos[env.robots[0].eef_site_id]
    cube_pos = env.sim.data.body_xpos[env.cube_body_id]
    goal_pos = env.goal_pos_world()

    dist_eef_cube = np.linalg.norm(eef_pos - cube_pos)
    dist_cube_goal = np.linalg.norm(cube_pos - goal_pos)

    reward_touch = 1.0 if env.check_contact_cube() else 0.0
    reward_dist_cube = -dist_eef_cube
    reward_dist_goal = -dist_cube_goal
    reward_avoid_table = -1.0 if env.check_contact_table() else 0.0

    if env._check_success():  
        return terminal_reward

    if dist_eef_cube > max_dist:
        return -terminal_reward

    reward = reward_touch + reward_dist_goal + reward_dist_cube + reward_avoid_table
    return reward