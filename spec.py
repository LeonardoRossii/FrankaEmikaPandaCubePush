def get_reward(env, action, lambdas):
    import numpy as np
    rewards = []
    for lam in lambdas:
        # Gather key states
        eef_pos = env.get_eef_pos()
        cube_pos = env.get_cube_pos()
        goal_pos = env.get_goal_pos()
        table_z = env.model.mujoco_arena.table_offset[2]

        d_eef_cube = float(np.linalg.norm(eef_pos - cube_pos))
        d_cube_goal = float(np.linalg.norm(goal_pos - cube_pos))

        # Weights for task shaping terms (also used to compute max_rewards_bonuses)
        w_reach = 1.0
        w_goal = 1.5
        w_align = 0.5
        w_c1 = 0.2
        w_c2 = 0.2
        w_both = 0.6
        w_height = 0.5

        # Task shaping terms
        reach_term = w_reach * (1.0 - np.tanh(4.0 * d_eef_cube))
        goal_term = w_goal * (1.0 - np.tanh(4.0 * d_cube_goal))

        # Alignment: encourage eef to be behind the cube relative to goal (push direction)
        v = goal_pos - cube_pos
        v_norm = np.linalg.norm(v) + 1e-8
        u = eef_pos - cube_pos
        u_norm = np.linalg.norm(u) + 1e-8
        cos = np.clip(np.dot(u, v) / (u_norm * v_norm), -1.0, 1.0)
        align_term = w_align * 0.5 * (1.0 - cos)  # max when u opposite to v

        # Finger contact shaping
        c1 = 1.0 if env.check_contact_finger_1_cube() else 0.0
        c2 = 1.0 if env.check_contact_finger_2_cube() else 0.0
        contact_term = w_c1 * c1 + w_c2 * c2 + w_both * (1.0 if (c1 > 0.0 and c2 > 0.0) else 0.0)

        # Vertical placement for safe pushing slightly above cube center
        target_z = cube_pos[2] + 0.005
        height_term = w_height * (1.0 - np.tanh(25.0 * abs(eef_pos[2] - target_z)))

        # Sum task-associated shaping terms
        task_reward = reach_term + goal_term + align_term + contact_term + height_term

        # Irreversible events dense penalty shaping
        irr_events_reward = 0.0

        # 1 & 2) Near table edge / unreachable region: penalize proximity to Y-bound
        try:
            bound_dist = float(env.get_cube_bound_dist())
        except Exception:
            bound_dist = 1.0
        irr_events_reward += -1.0 * np.exp(-5.0 * max(0.0, bound_dist))

        # 1) Cube falling off table (z below tabletop)
        drop = (cube_pos[2] - table_z) < 0.0
        irr_events_reward += -5.0 * (1.0 if drop else 0.0)

        # 3 & 8) Gripper-table contact as proxy for unsafe strikes or bumping table
        contact_table = env.check_contact_table()
        irr_events_reward += -1.0 * (1.0 if contact_table else 0.0)

        # 6) Finger wedged under cube: penalize eef below cube when close in XY
        eef_xy = eef_pos[:2]
        cube_xy = cube_pos[:2]
        dist_xy = float(np.linalg.norm(eef_xy - cube_xy))
        if dist_xy < 0.05:
            below = max(0.0, cube_pos[2] - eef_pos[2])
            irr_events_reward += -5.0 * np.tanh(20.0 * below)

        # Terminal reward term (success bonus)
        max_rewards_bonuses = w_reach + w_goal + w_align + w_c1 + w_c2 + w_both + w_height
        try:
            task_reward += 10 * env.horizon * max_rewards_bonuses * int(env.check_success(env))
        except TypeError:
            try:
                task_reward += 10 * env.horizon * max_rewards_bonuses * int(env.check_success())
            except Exception:
                success_local = (d_cube_goal < 0.04)
                task_reward += 10 * env.horizon * max_rewards_bonuses * int(success_local)

        # Final convex combination
        reward = lam * task_reward + (1.0 - lam) * irr_events_reward
        rewards.append(float(reward))

    return rewards


def get_success_condition(env):
    import numpy as np
    cube_pos = env.get_cube_pos()
    goal_pos = env.get_goal_pos()
    return float(np.linalg.norm(goal_pos - cube_pos)) < 0.04


def get_failure_condition(env):
    import numpy as np
    eef_pos = env.get_eef_pos()
    cube_pos = env.get_cube_pos()
    table_z = env.model.mujoco_arena.table_offset[2]

    d_eef_cube = float(np.linalg.norm(eef_pos - cube_pos))
    drop = (cube_pos[2] - table_z) < 0.0

    # Near/unreachable region at table edge (Y-bound approx.)
    try:
        bound_dist = float(env.get_cube_bound_dist())
    except Exception:
        bound_dist = 1.0
    near_edge = bound_dist < 0.005

    # Treat any irreversible event proxy as failure trigger
    irr_event = drop or near_edge

    failure = (d_eef_cube > 0.2)
    return bool(failure)