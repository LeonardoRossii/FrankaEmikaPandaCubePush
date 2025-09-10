import numpy as np

def get_reward(env, action):
    # Extract necessary positions
    eef_pos = env.sim.data.site_xpos[env.robots[0].eef_site_id]
    cube_pos = env.sim.data.body_xpos[env.cube_body_id]
    goal_pos = np.array([env.goal_xy[0], env.goal_xy[1], env.model.mujoco_arena.table_offset[2] + 0.001], dtype=float)
    
    # Calculate distances
    distance_to_cube = np.linalg.norm(eef_pos - cube_pos)
    distance_to_goal = np.linalg.norm(cube_pos - goal_pos)
    
    # Shaping rewards
    reach_reward = -distance_to_cube
    push_reward = -distance_to_goal if distance_to_cube < 0.05 else 0.0
    table_penalty = -1.0 if env.check_contact_table() else 0.0
    
    # Terminal reward
    terminal_reward = 1000.0 if np.linalg.norm(cube_pos - goal_pos) < 0.05 else 0.0
    
    # Failure penalty
    failure_penalty = -100.0 if distance_to_cube > 0.2 else 0.0
    
    # Total reward
    reward = reach_reward + push_reward + table_penalty + terminal_reward + failure_penalty
    return reward

def get_success_condition(env):
    cube_pos = env.sim.data.body_xpos[env.cube_body_id]
    goal_pos = np.array([env.goal_xy[0], env.goal_xy[1], env.model.mujoco_arena.table_offset[2] + 0.001], dtype=float)
    success = np.linalg.norm(cube_pos - goal_pos) < 0.05
    return success

def get_failure_condition(env):
    eef_pos = env.sim.data.site_xpos[env.robots[0].eef_site_id]
    cube_pos = env.sim.data.body_xpos[env.cube_body_id]
    failure = np.linalg.norm(eef_pos - cube_pos) > 0.2
    return failure