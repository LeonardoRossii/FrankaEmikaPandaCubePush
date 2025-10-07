def get_reward(env, _):
    import numpy as np

    lam = 0.5
    reward = 0.0

    # Poses and useful quantities
    eef = env.get_eef_pos().astype(float)
    cube = env.get_cube_pos().astype(float)
    goal = env.get_goal_pos().astype(float)
    table_z = float(env.model.mujoco_arena.table_offset[2])
    table_size = env.table_full_size

    d_eef_cube = float(np.linalg.norm(eef - cube))
    d_cube_goal = float(np.linalg.norm(cube - goal))

    # Unit vectors for alignment and velocity shaping
    dir_cg = goal - cube
    norm_cg = np.linalg.norm(dir_cg)
    u_cg = dir_cg / norm_cg if norm_cg > 1e-8 else np.zeros(3)

    dir_ec = cube - eef
    norm_ec = np.linalg.norm(dir_ec)
    u_ec = dir_ec / norm_ec if norm_ec > 1e-8 else np.zeros(3)

    # 1) Reach the cube
    r_reach = 1.0 - np.tanh(4.0 * d_eef_cube)  # in (0,1)

    # 2) Contacts with both fingers
    c1 = 1.0 if env.check_contact_finger_1_cube() else 0.0
    c2 = 1.0 if env.check_contact_finger_2_cube() else 0.0
    both = 1.0 if (c1 > 0.5 and c2 > 0.5) else 0.0
    r_contacts = 0.3 * c1 + 0.3 * c2 + 0.7 * both  # max 1.3

    # 3) Good push alignment: eef behind cube relative to goal direction
    r_align = float(max(0.0, np.dot(u_ec, u_cg)))  # in [0,1]

    # 4) Move cube toward goal (position proximity)
    r_goal = 1.0 - np.tanh(4.0 * d_cube_goal)  # in (0,1)

    # Weights for task terms
    w_reach = 1.0
    w_contacts = 1.0  # already internally weighted to max 1.3
    w_align = 0.5
    w_goal = 2.0
    w_vel = 0.5

    # Sum task reward
    reward += (
        w_reach * r_reach
        + w_contacts * r_contacts
        + w_align * r_align
        + w_goal * r_goal
    )

    # Compute max possible sum of positive bonuses (for terminal term)
    max_rewards_bonuses = (
        w_reach * 1.0
        + 1.3  # r_contacts maximum
        + w_align * 1.0
        + w_goal * 1.0
        + w_vel * 1.0
    )

    # Irreversible-events dense penalties
    irr_events_reward = 0.0

    # 1-2,8) Cube near table boundary or outside reachable region
    # Distance to table boundaries in x,y
    half_x = table_size[0] / 2.0
    half_y = table_size[1] / 2.0
    dist_x = max(0.0, half_x - abs(cube[0]))
    dist_y = max(0.0, half_y - abs(cube[1]))
    boundary_dist = min(dist_x, dist_y)
    # Penalize when closer than 5 cm to any edge
    edge_thresh = 0.05
    if boundary_dist < edge_thresh:
        edge_risk = (edge_thresh - boundary_dist) / edge_thresh  # in (0,1]
        irr_events_reward += -0.5 * edge_risk

    # 3) Gripper / arm contact with table; also penalize dangerously low eef height
    if env.check_contact_table():
        irr_events_reward += -0.3
    min_clearance = 0.01
    clearance = eef[2] - (table_z + min_clearance)
    if clearance < 0.0:
        irr_events_reward += -0.2 * min(1.0, abs(clearance) / min_clearance)

    # 5) Finger jams under cube: eef too low under cube when close
    cube_half_h = 0.02  # from object definition
    cube_bottom_z = cube[2] - cube_half_h
    if d_eef_cube < 0.07:
        under_depth = cube_bottom_z - eef[2]  # >0 means eef under bottom
        if under_depth > 0.0:
            irr_events_reward += -0.3 * min(1.0, under_depth / 0.02)

    # 6) Excessive force (proxy): use safety filter effort if available
    eff = getattr(env, "safety_filter_effort", 0.0)
    if eff is not None and eff > 0.0:
        irr_events_reward += -0.3 * min(1.0, eff / 10.0)

    # 8) Cube pushed too far from reachable workspace: eef too far from cube
    if d_eef_cube > 0.18:
        far_risk = min(1.0, (d_eef_cube - 0.18) / 0.12)  # ramps until ~0.30m
        irr_events_reward += -0.3 * far_risk

    # Terminal reward on success
    reward += 10 * env.horizon * max_rewards_bonuses * int(get_success_condition(env))

    # Convex combination between task reward and irreversible-events term
    total_reward = lam * reward + (1.0 - lam) * irr_events_reward
    return float(total_reward)


def get_success_condition(env):
    import numpy as np

    cube = env.get_cube_pos().astype(float)
    goal = env.get_goal_pos().astype(float)
    d = float(np.linalg.norm(cube - goal))
    return d < 0.04


def get_failure_condition(env):
    import numpy as np

    eef = env.get_eef_pos().astype(float)
    cube = env.get_cube_pos().astype(float)
    d = float(np.linalg.norm(eef - cube))
    return d > 0.2