import numpy as np

def get_reward(env, action):
    reward = 0.0

    if env._check_success():
        reward = 10000

    elif env.reward_shaping:
        
        eef_pos = env.sim.data.site_xpos[env.robots[0].eef_site_id]
        cube_pos = env.sim.data.body_xpos[env.cube_body_id]      
        goal_pos = env.goal_pos_world()                       

        reach_dist = np.linalg.norm(eef_pos - cube_pos)
        reach_r = 1.0 - np.tanh(10.0 * reach_dist)
        reward += 0.5 * reach_r

        obj_goal_dist = np.linalg.norm(cube_pos[:2] - goal_pos[:2])
        if not hasattr(env, "_last_obj_goal_dist"):
            env._last_obj_goal_dist = obj_goal_dist
        progress = env._last_obj_goal_dist - obj_goal_dist
        reward += 1.0 * progress
        env._last_obj_goal_dist = obj_goal_dist

        table_z = env.model.mujoco_arena.table_offset[2]
        lift_pen = max(0.0, cube_pos[2] - (table_z + 0.01))
        reward -= 0.25 * lift_pen

        if env.check_contact(env.robots[0].gripper, env.cube):
            reward += 0.25
        
        if env.check_contact_table():
            reward -= 0.2
    
    if env.reward_scale is not None:
        reward *= env.reward_scale / 20

    return float(reward)