import numpy as np

def get_reward(env, _, lambdas):
    rewards = []
    # Common state
    eef = env.get_eef_pos().copy()
    cube = env.get_cube_pos().copy()
    goal = env.get_goal_pos().copy()

    table_z = env.model.mujoco_arena.table_offset[2]
    table_size = np.array(env.table_full_size)  # (x_len, y_len, z_thick)
    half_x = table_size[0] / 2.0
    half_y = table_size[1] / 2.0

    # Distances
    d_eef_cube = np.linalg.norm(eef - cube)
    d_cg = np.linalg.norm(cube - goal)

    # Shaping helpers
    def smooth_close(dist, scale=5.0):
        # maps dist >=0 to (0,1), higher when closer
        return 1.0 - np.tanh(scale * dist)

    # Positive Task Terms
    # 1) Reach the cube
    reach_w = 1.0
    reach_r = reach_w * smooth_close(d_eef_cube, scale=4.0)

    # 2) Goal proximity (cube to goal)
    goal_w = 1.0
    goal_r = goal_w * smooth_close(d_cg, scale=4.0)

    # 3) Two-finger contact on cube
    f1 = 1 if env.check_contact_finger_1_cube() else 0
    f2 = 1 if env.check_contact_finger_2_cube() else 0
    contact_w = 1.0
    contact_base = 0.4 * f1 + 0.4 * f2 + 0.2 * int(f1 and f2)  # max 1.0
    contact_r = contact_w * contact_base

    # 4) Directional alignment for pushing (eef behind cube wrt goal)
    dir_w = 0.5
    v_cef = eef - cube
    v_cg = goal - cube
    cos_term = 0.0
    if np.linalg.norm(v_cef) > 1e-6 and np.linalg.norm(v_cg) > 1e-6:
        v_cef_n = v_cef / (np.linalg.norm(v_cef) + 1e-8)
        v_opp_goal = -v_cg / (np.linalg.norm(v_cg) + 1e-8)
        cos_term = np.clip(np.dot(v_cef_n, v_opp_goal), -1.0, 1.0)
    # Gate directional reward to when we're close to the cube
    gate = np.clip(1.0 - np.tanh(10.0 * (d_eef_cube - 0.06)), 0.0, 1.0)
    dir_r = dir_w * (0.5 * (cos_term + 1.0)) * gate  # maps cos [-1,1] -> [0,1]

    # 5) Vertical alignment (keep eef roughly at cube height to push without lifting/scraping)
    height_w = 0.5
    dz = abs(eef[2] - cube[2])
    height_r = height_w * smooth_close(dz, scale=20.0)

    # Sum task reward terms
    base_task_reward = reach_r + goal_r + contact_r + dir_r + height_r

    # Compute max of positive bonuses for terminal scaling (exact sum of maxima)
    max_rewards_bonuses = reach_w * 1.0 + goal_w * 1.0 + contact_w * 1.0 + dir_w * 1.0 + height_w * 1.0

    # Irreversible events shaping (dense penalties; negative or zero)
    irr_penalties = []

    # 1) Cube off table / falling risk: proximity to table edges (x and y) and drop
    safety_margin = 0.05  # 5 cm
    cube_x_margin = half_x - abs(cube[0])
    cube_y_margin = half_y - abs(cube[1])
    # edge proximity penalty (0 when far, -1 when at or beyond edge)
    cube_edge_pen_x = -np.clip((safety_margin - cube_x_margin) / max(safety_margin, 1e-6), 0.0, 1.0)
    cube_edge_pen_y = -np.clip((safety_margin - cube_y_margin) / max(safety_margin, 1e-6), 0.0, 1.0)
    irr_penalties.append(0.5 * (cube_edge_pen_x + cube_edge_pen_y))
    # drop penalty
    cube_drop = (cube[2] - table_z) < -0.01
    irr_penalties.append(-1.0 if cube_drop else 0.0)

    # 2) Gripper-table collision risk
    eef_clear = eef[2] - table_z
    eef_low_clear_pen = -np.clip((0.01 - eef_clear) / 0.01, 0.0, 1.0)  # penalize when < 1cm clearance
    irr_penalties.append(0.5 * eef_low_clear_pen)
    table_contact = env.check_contact_table()
    irr_penalties.append(-0.5 if table_contact else 0.0)

    # 3) Arm self-collision / 7) Joint limits (proxy: safety_filter_effort)
    sf_effort = float(getattr(env, "safety_filter_effort", 0.0))
    irr_penalties.append(-0.5 * np.clip(sf_effort, 0.0, 1.0))

    # 4) Strike scene boundary (proxy: eef near horizontal workspace edges)
    eef_x_margin = half_x - abs(eef[0])
    eef_y_margin = half_y - abs(eef[1])
    eef_edge_pen_x = -np.clip((safety_margin - eef_x_margin) / max(safety_margin, 1e-6), 0.0, 1.0)
    eef_edge_pen_y = -np.clip((safety_margin - eef_y_margin) / max(safety_margin, 1e-6), 0.0, 1.0)
    irr_penalties.append(0.3 * (eef_edge_pen_x + eef_edge_pen_y))

    # 5) Gripper jams (proxy: both fingers on cube very low above table)
    jam_like = int(f1 and f2 and (eef_clear < 0.008))
    irr_penalties.append(-0.5 * jam_like)

    # 6) Cube pushed outside reachable/workspace (proxy: beyond table bounds with small margin)
    cube_out_x = abs(cube[0]) > (half_x + 0.02)
    cube_out_y = abs(cube[1]) > (half_y + 0.02)
    irr_penalties.append(-0.7 if (cube_out_x or cube_out_y) else 0.0)

    # 8) Unstable / invalid states
    invalid_state = (not np.isfinite(eef).all()) or (not np.isfinite(cube).all())
    irr_penalties.append(-1.0 if invalid_state else 0.0)

    irr_events_reward = float(np.sum(irr_penalties))

    # Terminal success bonus
    success = get_success_condition(env)
    base_task_reward += 10.0 * env.horizon * max_rewards_bonuses * int(success)

    # Mix with lambda weights
    for lam in lambdas:
        lam = float(np.clip(lam, 0.0, 1.0))
        reward = lam * base_task_reward + (1.0 - lam) * irr_events_reward
        rewards.append(float(reward))

    return rewards

def get_success_condition(env):
    cube = env.get_cube_pos().copy()
    goal = env.get_goal_pos().copy()
    success = np.linalg.norm(cube - goal) < 0.04
    return bool(success)

def get_failure_condition(env):
    # Distance-based failure
    eef = env.get_eef_pos().copy()
    cube = env.get_cube_pos().copy()
    d_eef_cube = np.linalg.norm(eef - cube)
    far_from_cube = d_eef_cube > 0.2

    # Irreversible events detection
    table_z = env.model.mujoco_arena.table_offset[2]
    table_size = np.array(env.table_full_size)
    half_x = table_size[0] / 2.0
    half_y = table_size[1] / 2.0

    # 1) Cube dropped off table
    cube_drop = (cube[2] - table_z) < -0.01

    # 2) Gripper-table collision with excessive force (proxy)
    eef_clear = eef[2] - table_z
    table_contact = env.check_contact_table()
    hard_table_hit = table_contact and (eef_clear < 0.005)

    # 3) Self-collision (proxy via safety filter effort)
    sf_effort = float(getattr(env, "safety_filter_effort", 0.0))
    self_collision_like = sf_effort > 0.7

    # 4) Strike scene boundary (eef beyond bounds + margin)
    eef_out = (abs(eef[0]) > (half_x + 0.05)) or (abs(eef[1]) > (half_y + 0.05)) or (eef[2] < table_z - 0.01)

    # 5) Jam (proxy: both fingers on cube and eef very low)
    f1 = env.check_contact_finger_1_cube()
    f2 = env.check_contact_finger_2_cube()
    jam = f1 and f2 and (eef_clear < 0.005)

    # 6) Cube outside workspace (beyond table bounds + margin)
    cube_out = (abs(cube[0]) > (half_x + 0.02)) or (abs(cube[1]) > (half_y + 0.02))

    # 7) Joint limits (proxy via safety filter effort)
    joint_limit_like = sf_effort > 0.7

    # 8) Unstable control / invalid states
    invalid_state = (not np.isfinite(eef).all()) or (not np.isfinite(cube).all())

    irreversible = cube_drop or hard_table_hit or self_collision_like or eef_out or jam or cube_out or joint_limit_like or invalid_state

    failure = bool(far_from_cube or irreversible)
    return failure