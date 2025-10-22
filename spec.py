def get_reward(env, _, lambdas):
    import numpy as np

    # Default lambda list if not provided
    if lambdas is None or len(lambdas) == 0:
        lambdas = [0.9]

    rewards = []
    # Gather key states
    eef = env.get_eef_pos()
    cube = env.get_cube_pos()
    goal = env.get_goal_pos()
    table_z = env.model.mujoco_arena.table_offset[2]

    # Distances
    d_ec = float(np.linalg.norm(eef - cube))  # end-effector to cube (3D)
    d_cg_xy = float(np.linalg.norm((cube - goal)[:2]))  # cube to goal (XY)

    # Contacts
    contact_f1 = 1.0 if env.check_contact_finger_1_cube() else 0.0
    contact_f2 = 1.0 if env.check_contact_finger_2_cube() else 0.0

    # Directional alignment (EEF behind cube, pushing toward goal)
    u = cube[:2] - eef[:2]
    v = goal[:2] - cube[:2]
    u_norm = np.linalg.norm(u) + 1e-8
    v_norm = np.linalg.norm(v) + 1e-8
    align = np.dot(u, v) / (u_norm * v_norm)
    align = float(np.clip(align, 0.0, 1.0))  # only reward good alignment

    # Positive shaping terms and weights (max of each term is 1.0)
    w_push = 1.5
    w_reach = 1.0
    w_align = 0.7
    w_touch = 0.4  # per-finger

    r_push = 1.0 - np.tanh(4.0 * d_cg_xy)         # in [0, 1)
    r_reach = 1.0 - np.tanh(4.0 * d_ec)           # in [0, 1)
    r_align = align                                # in [0, 1]
    r_touch = contact_f1 + contact_f2              # in {0,1,2}

    # Task reward
    reward_task = (
        w_push * r_push
        + w_reach * r_reach
        + w_align * r_align
        + w_touch * r_touch
    )

    # Terminal success bonus (as specified)
    max_rewards_bonuses = w_push + w_reach + w_align + 2.0 * w_touch  # equals 4.0 with current weights
    reward_task += 10 * env.horizon * max_rewards_bonuses * int(env.check_success())

    # Irreversible event dense penalties (irr_events_reward)
    # - Cube fall/drop proximity and event
    cube_drop = (cube[2] - table_z) < -0.025
    # Risk near lateral table Y-edge using provided boundary distance (only Y available)
    edge_margin_y = 0.05  # meters
    try:
        cube_bound_dist_y = float(env.get_cube_bound_dist())
    except Exception:
        cube_bound_dist_y = edge_margin_y
    risk_edge_y = float(np.clip(1.0 - cube_bound_dist_y / max(edge_margin_y, 1e-6), 0.0, 1.0))

    # - Safety filter efforts
    filt_robot_table = float(np.clip(env.get_robot_table_collision_avoidance_safety_filter_effort(), 0.0, 1.0))
    filt_cube_drop = float(np.clip(env.get_cube_drop_off_table_avoidance_safety_filter_effort(), 0.0, 1.0))

    # - Table contact (proxy for unsafe interactions)
    table_contact = env.check_contact_table()

    # Weights for irreversible penalties (keep magnitudes modest)
    w_drop_flag = 1.0
    w_edge_risk = 0.3
    w_robot_table_effort = 0.5
    w_cube_drop_effort = 0.5
    w_table_contact = 0.2

    irr_events_reward = 0.0
    irr_events_reward += -w_drop_flag * (1.0 if cube_drop else 0.0)
    irr_events_reward += -w_edge_risk * risk_edge_y
    irr_events_reward += -w_robot_table_effort * filt_robot_table
    irr_events_reward += -w_cube_drop_effort * filt_cube_drop
    irr_events_reward += -w_table_contact * (1.0 if table_contact else 0.0)

    # Build final reward for each lambda (convex combination)
    for lam in lambdas:
        lam = float(np.clip(lam, 0.0, 1.0))
        reward = reward_task
        reward = lam * reward + (1.0 - lam) * irr_events_reward
        rewards.append(float(reward))

    return rewards


def get_success_condition(env):
    import numpy as np
    cube = env.get_cube_pos()
    goal = env.get_goal_pos()
    d_cg_xy = float(np.linalg.norm((cube - goal)[:2]))
    success = d_cg_xy < 0.04
    return bool(success)


def get_failure_condition(env):
    import numpy as np
    # Distance-based failure
    eef = env.get_eef_pos()
    cube = env.get_cube_pos()
    table_z = env.model.mujoco_arena.table_offset[2]

    d_ec = float(np.linalg.norm(eef - cube))
    far_from_cube = d_ec > 0.2

    # Irreversible events (detectable proxies)
    cube_drop = (cube[2] - table_z) < -0.025
    filt_robot_table = float(env.get_robot_table_collision_avoidance_safety_filter_effort())
    filt_cube_drop = float(env.get_cube_drop_off_table_avoidance_safety_filter_effort())

    # Thresholds for safety filter efforts indicating likely protective stops
    robot_table_fault = filt_robot_table > 0.5
    cube_drop_fault = filt_cube_drop > 0.5

    irreversible = cube_drop

    failure = bool(far_from_cube or irreversible)
    return failure