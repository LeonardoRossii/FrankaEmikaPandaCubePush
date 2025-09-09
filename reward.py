import numpy as np

def get_reward(env, action):
    # Constants
    max_dist = 0.2
    T = 250
    terminal_reward = 1000  # A large reward for task completion

    # Positions
    eef_pos = env.sim.data.site_xpos[env.robots[0].eef_site_id]
    cube_pos = env.sim.data.body_xpos[env.cube_body_id]
    goal_pos = env.goal_pos_world()

    # Distance calculations
    dist_eef_cube = np.linalg.norm(eef_pos - cube_pos)
    dist_cube_goal = np.linalg.norm(cube_pos - goal_pos)

    # Shaping rewards
    reward_touch = 1.0 if env.check_contact(env.robots[0].gripper, env.cube) else 0.0
    reward_push = -dist_cube_goal  # Encourage moving the cube towards the goal
    reward_avoid_table = -1.0 if env.check_contact_table() else 0.0

    # Terminal reward
    if dist_cube_goal < 0.05:  
        return terminal_reward

    # Failure condition
    if dist_eef_cube > max_dist:
        return -terminal_reward

    # Total reward
    reward = reward_touch + reward_push + reward_avoid_table
    return reward