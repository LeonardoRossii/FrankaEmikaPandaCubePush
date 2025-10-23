import numpy as np
import scipy.sparse as sp
import osqp


# ------------------------------ Base CBF module ------------------------------
class CBFModule:
    def constraints(self, _):
        """
        Return:
            As: list[np.ndarray shape (n_joints,)]
            bs: list[float]
        representing inequalities a_i @ qdot <= b_i
        """
        raise NotImplementedError


# ------------------------------ Table plane CBF ------------------------------
class TableTopCBF(CBFModule):
    def __init__(self, env):
        self.env = env
        self.cz = float(self.env.model.mujoco_arena.table_offset[2])
        self.robot_config = {
            "bodies": {
                "gripper0_eef":        {"alpha": 2.0, "margin": 0.01},
                "gripper0_leftfinger": {"alpha": 2.0, "margin": 0.01},
                "gripper0_rightfinger":{"alpha": 2.0, "margin": 0.01},
                "robot0_link7":        {"alpha": 1.0, "margin": 0.05},
                "robot0_link6":        {"alpha": 1.0, "margin": 0.05},
                "robot0_link5":        {"alpha": 1.0, "margin": 0.05},
                "robot0_link4":        {"alpha": 1.0, "margin": 0.05},
                "robot0_link3":        {"alpha": 1.0, "margin": 0.05},
            },
            "geoms": {
                "robot0_link7_collision": {"alpha": 1.0, "margin": 0.05},
                "robot0_link6_collision": {"alpha": 1.0, "margin": 0.05},
                "robot0_link5_collision": {"alpha": 1.0, "margin": 0.05},
                "robot0_link4_collision": {"alpha": 1.0, "margin": 0.05},
                "robot0_link3_collision": {"alpha": 1.0, "margin": 0.05},
            }
        }

    @staticmethod
    def _table_plane_cbf(env, p, margin):
        z_top = float(env.model.mujoco_arena.table_offset[2])
        h = float(p[2]) - (z_top + margin)
        H = np.array([0.0, 0.0, 1.0], dtype=float)
        return h, H

    def constraints(self, env):
        As, bs = [], []
        sim = env.sim
        model = sim.model
        robot_joint_idx = env.robots[0].joint_indexes

        # Bodies
        for body_name, params in self.robot_config["bodies"].items():
            alpha = float(params["alpha"]); margin = float(params["margin"])
            x = sim.data.body_xpos[model.body_name2id(body_name)][:3]
            if abs(x[2] - self.cz) < 10.0:
                h, H = self._table_plane_cbf(env, x, margin)
                jacp = np.reshape(sim.data.get_body_jacp(body_name), (3, -1))
                jacr = np.reshape(sim.data.get_body_jacr(body_name), (3, -1))
                J_full = np.vstack((jacp, jacr))
                J_robot_pos = J_full[:3, robot_joint_idx]
                a = (H @ J_robot_pos).ravel()
                As.append(a.astype(float))
                bs.append(float(-alpha * h))

        # Geoms
        for geom_name, params in self.robot_config["geoms"].items():
            alpha = float(params["alpha"]); margin = float(params["margin"])
            x = sim.data.geom_xpos[model.geom_name2id(geom_name)][:3]
            if abs(x[2] - self.cz) < 10.0:
                h, H = self._table_plane_cbf(env, x, margin)
                jacp = np.reshape(sim.data.get_geom_jacp(geom_name), (3, -1))
                jacr = np.reshape(sim.data.get_geom_jacr(geom_name), (3, -1))
                J_full = np.vstack((jacp, jacr))
                J_robot_pos = J_full[:3, robot_joint_idx]
                a = (H @ J_robot_pos).ravel()
                As.append(a.astype(float))
                bs.append(float(-alpha * h))
        return As, bs


# -------------------------- Cube boundary (drop) CBF -------------------------
class CubeDropCBF(CBFModule):
    """
    Gripper-site-only version (no contact geoms).
    Keeps the cube on the table with a planar CBF and adds hard constraints
    on end-effector angular velocity to avoid peel-off.
    """
    def __init__(self, env):
        self.env = env
        self.cfg = {
            "alpha": 0.25,
            "margin": 0.01,
            "rx": 0.020,
            "ry": 0.020,
            "e1": 0.1,
            "e2": 0.1,
            "kappa": 1e-12,
            # wrist rotation limits
            "omega_max": 0.2,   # rad/s
            "limit_axes": "z",  # "z" or "all"
        }

    # Barrier h(x,y)
    def bf(self):
        cfg = self.cfg
        cx, cy = 0.0, 0.0
        Lx = self.env.table_full_size[0] / 2.0
        Ly = self.env.table_full_size[1] / 2.0
        mx, my = Lx - cfg["rx"], Ly - cfg["ry"]

        p = self.env.sim.data.body_xpos[self.env.cube_body_id]
        xc, yc = float(p[0]), float(p[1])

        e1, e2, kappa = cfg["e1"], cfg["e2"], cfg["kappa"]
        ux, uy = (xc - cx) / mx, (yc - cy) / my
        ax, ay = np.sqrt(ux * ux + kappa), np.sqrt(uy * uy + kappa)

        Sx = ax ** (2.0 / e2)
        Sy = ay ** (2.0 / e2)
        S = Sx + Sy
        F = S ** (e2 / e1)
        h = 1.0 - (F - cfg["margin"])

        sgnx, sgny = ux / ax, uy / ay
        dSx_dxc = (2.0 / e2) * (ax ** (2.0 / e2 - 1.0)) * sgnx / mx
        dSy_dyc = (2.0 / e2) * (ay ** (2.0 / e2 - 1.0)) * sgny / my
        c = (e2 / e1) * (S ** (e2 / e1 - 1.0)) if S > 0 else 0.0
        dh_dxc = -c * dSx_dxc
        dh_dyc = -c * dSy_dyc
        H_xy = np.array([dh_dxc, dh_dyc], dtype=float)  # (2,)
        return float(h), H_xy

    # CBF row using gripper site only
    def _cube_drop_row_gripper_site(self):
        h, H_xy = self.bf()
        sim = self.env.sim
        qvel_idx = self.env.robots[0].joint_indexes

        ee_site = self.env.robots[0].gripper.important_sites["grip_site"]

        # site Jacobians (world frame)
        jacp = np.reshape(sim.data.get_site_jacp(ee_site), (3, -1))
        jacr = np.reshape(sim.data.get_site_jacr(ee_site), (3, -1))

        # positional block for robot joints, planar slice
        J_full = np.vstack((jacp, jacr))           # (6 x nv)
        J_robot_pos = J_full[:3, qvel_idx]         # (3 x n)
        J_xy = J_robot_pos[:2, :]                  # (2 x n)

        # CBF row: dot(h) = H_xy^T [v_x; v_y] ≈ H_xy^T J_xy qdot ≥ -α h
        a = (H_xy.reshape(1, 2) @ J_xy).ravel()
        b = -float(self.cfg["alpha"]) * float(h)
        return a.astype(float), float(b)

    # Hard bounds on EEF angular velocity at the gripper site
    def _wrist_rotation_limit_rows(self):
        """
        Encode |omega_axis| <= omega_max as A u >= b with:
            (+row) @ qdot >= -omega_max
            (-row) @ qdot >= -omega_max
        """
        sim = self.env.sim
        qvel_idx = self.env.robots[0].joint_indexes
        omega_max = float(self.cfg.get("omega_max", 0.2))
        limit_axes = self.cfg.get("limit_axes", "z")  # "z" or "all"

        ee_site = self.env.robots[0].gripper.important_sites["grip_site"]
        Jr_site = np.reshape(sim.data.get_site_jacr(ee_site), (3, -1))
        Jr = Jr_site[:, qvel_idx]  # (3 x n)

        axes = [2] if limit_axes == "z" else [0, 1, 2]
        As, Bs = [], []
        for ax in axes:
            row = Jr[ax, :].astype(float)  # omega_ax = row @ qdot
            As.append(+row)               # +row @ qdot >= -omega_max
            Bs.append(-omega_max)
            As.append(-row)               # -row @ qdot >= -omega_max (i.e., row @ qdot <= +omega_max)
            Bs.append(-omega_max)
        return As, Bs

    def constraints(self, env):
        self.env = env
        As, Bs = [], []
        try:
            a_cd, b_cd = self._cube_drop_row_gripper_site()
            As.append(a_cd); Bs.append(b_cd)

            rot_As, rot_Bs = self._wrist_rotation_limit_rows()
            As.extend(rot_As); Bs.extend(rot_Bs)
        except Exception:
            return [], []
        return As, Bs


# ------------------------------- QP filter -----------------------------------
class CollisionQPFilter:
    def __init__(self, env, cbf_modules):
        self.env = env
        self.modules = cbf_modules

    @staticmethod
    def _solve_qp(u_des, a_list, b_list, *, drop_tol=1e-10, normalize_rows=True):
        """
        Solve:  minimize 0.5||u - u_des||^2  s.t.  A u >= b
        Always returns (u, y, ok). y has length m (kept rows).
        """
        u_des = np.asarray(u_des, dtype=float).reshape(-1)
        n = u_des.size

        # ---- sanitize constraints ----
        A_clean, b_clean = [], []
        for a, b in zip(a_list, b_list):
            a = np.asarray(a, dtype=float).reshape(n)
            b = float(b)
            if not np.all(np.isfinite(a)) or not np.isfinite(b):
                continue
            norm = np.linalg.norm(a)
            if norm < drop_tol:
                # near-zero row: drop (redundant if b<=0; infeasible if b>0 — drop anyway)
                continue
            if normalize_rows:
                a = a / norm
                b = b / norm
            A_clean.append(a)
            b_clean.append(b)

        m = len(A_clean)
        if m == 0:
            # No constraints -> nominal, zero duals
            return u_des, np.zeros(0, dtype=float), True

        A_mat = np.stack(A_clean, axis=0)                  # (m, n)
        b_arr = np.array(b_clean, dtype=float)             # (m,)

        # OSQP data for A u >= b  ↔  l=b, u=+inf
        Au = sp.csc_matrix(A_mat)
        l = b_arr
        u = np.full(m, np.inf, dtype=float)
        P = sp.eye(n, format="csc")
        q = -u_des

        prob = osqp.OSQP()
        prob.setup(P=P, q=q, A=Au, l=l, u=u,
                   verbose=False, polish=False, eps_abs=1e-6, eps_rel=1e-6)
        res = prob.solve()

        ok = (res.info.status_val in (1, 2)) and (res.x is not None)
        if not ok:
            return u_des, np.zeros(m, dtype=float), False

        x = np.array(res.x, dtype=float)
        y = np.array(res.y, dtype=float) if res.y is not None else np.zeros(m, dtype=float)
        return x, y, True

    def apply(self, u_des):
        """
        u_des: full action vector; first 7 entries are robot joint velocities
        Returns:
            u_act: safe action (same shape as u_des)
            efforts: dict with per-module effort diagnostics (optional)
        """
        u_act = np.array(u_des, dtype=float).copy()
        u_nom = u_act[:7]

        all_As, all_Bs = [], []
        groups = []
        row_start = 0

        for mod in self.modules:
            a_list, b_list = mod.constraints(self.env)
            k = len(a_list)
            if k:
                all_As.extend(a_list)
                all_Bs.extend(b_list)
                groups.append((mod.__class__.__name__, row_start, row_start + k))
                row_start += k

        # No constraints → keep nominal
        if len(all_As) == 0:
            return u_act, {"qp_skipped": True}

        u_safe, y, ok = self._solve_qp(u_nom, all_As, all_Bs)
        u_act[:7] = u_safe if ok else u_nom

        # Optional diagnostics
        efforts = {}
        if len(all_As) > 0:
            A_mat = np.stack([np.asarray(ai, dtype=float).reshape(-1) for ai in all_As], axis=0)
            for name, lo, hi in groups:
                A_g = A_mat[lo:hi, :]
                y_g = y[lo:hi] if y.size >= hi else np.zeros(hi - lo, dtype=float)
                delta_g = -A_g.T @ y_g
                efforts[name] = {
                    "delta": delta_g,
                    "effort_l2": float(np.linalg.norm(delta_g)),
                    "active_constraints": int(np.count_nonzero(y_g)),
                }
        return u_act, efforts
