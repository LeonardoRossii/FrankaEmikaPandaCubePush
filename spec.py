import numpy as np

def get_reward(env, action):
    # Extract necessary positions
    eef_pos = env.sim.data.site_xpos[env.robots[0].eef_site_id]
    cube_pos = env.sim.data.body_xpos[env.cube_body_id]
    goal_pos = np.array([env.goal_xy[0], env.goal_xy[1], env.model.mujoco_arena.table_offset[2] + 0.001], dtype=float)
    
    # Calculate distances
    eef_to_cube_dist = np.linalg.norm(eef_pos - cube_pos)
    cube_to_goal_dist = np.linalg.norm(cube_pos - goal_pos)
    
    # Shaping rewards
    reach_reward = 1.0 / (1.0 + eef_to_cube_dist)  # Encourage reaching the cube
    push_reward = 1.0 / (1.0 + cube_to_goal_dist)  # Encourage pushing the cube to the goal
    
    # Penalty for touching the table
    table_penalty = -1.0 if env.check_contact_table() else 0.0
    
    # Terminal reward
    terminal_reward_term = 10 * env.horizon * (1.0 + 1.0)  # Maximum possible shaping rewards
    
    # Check success and failure
    success = get_success_condition(env)
    failure = get_failure_condition(env)
    
    # Calculate reward
    reward = reach_reward + push_reward + table_penalty
    if success:
        reward += terminal_reward_term
    if failure:
        reward -= terminal_reward_term
    
    return reward

def get_success_condition(env):
    cube_pos = env.sim.data.body_xpos[env.cube_body_id]
    goal_pos = np.array([env.goal_xy[0], env.goal_xy[1], env.model.mujoco_arena.table_offset[2] + 0.001], dtype=float)
    success = np.linalg.norm(cube_pos - goal_pos) < 0.05  # Success if cube is close to the goal
    return success

def get_failure_condition(env):
    eef_pos = env.sim.data.site_xpos[env.robots[0].eef_site_id]
    cube_pos = env.sim.data.body_xpos[env.cube_body_id]
    failure = np.linalg.norm(eef_pos - cube_pos) > 0.2  # Failure if eef is too far from the cube
    return failure