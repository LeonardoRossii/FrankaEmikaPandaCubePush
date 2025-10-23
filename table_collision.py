import osqp
import numpy as np
import scipy.sparse as sp

class CBFModule:
    def constraints(self, _):
        raise NotImplementedError
    def objective_terms(self, _):
        return None

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

    def _table_plane_cbf(self,env, p, margin):
        z_top = float(env.model.mujoco_arena.table_offset[2])
        h = float(p[2]) - (z_top + margin)
        H = np.array([0.0, 0.0, 1.0], dtype=float)
        return h, H

    def constraints(self, env):
        As, bs = [], []
        sim = env.sim
        model = sim.model
        robot_joint_idx = env.robots[0].joint_indexes

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

"""class WristRateLimitCBF(CBFModule):
    def __init__(self, env, omega_max=0.5):
        self.env = env
        self.omega_max = omega_max

    def constraints(self, env):
        As, Bs = [], []
        qvel_idx = env.robots[0].joint_indexes
        ee_site = env.robots[0].gripper.important_sites["grip_site"]
        Jr_site = np.reshape(env.sim.data.get_site_jacr(ee_site), (3, -1))
        Jr = Jr_site[:, qvel_idx]
        axes = [0, 1, 2]
        for ax in axes:
            row = Jr[ax, :].astype(float)
            As.append(+row); Bs.append(-self.omega_max)
            As.append(-row); Bs.append(-self.omega_max)
        return As, Bs"""
    

class CubeDropCBF(CBFModule):
    def __init__(self, env):
        self.env = env
        self.cfg = {
            "alpha": 0.25,
            "margin": 0.02,
            "rx": 0.020,
            "ry": 0.020,
            "e1": 0.1,
            "e2": 0.1,
            "kappa": 1e-12,
            "omega_max": 0.5,
            "limit_axes": "all",
            "ori_hold_weight": 10.0,  # weight of the quadratic term
            "ori_kp": 2.0,            # gain from orientation error to omega_ref
            "ori_axes": "all",        # "z" or "all"
        }
        self._R0 = None

    @staticmethod
    def _mat_from_site(sim, site_name):
        xmat = np.array(sim.data.get_site_xmat(site_name), dtype=float).reshape(3, 3)
        return xmat

    @staticmethod
    def _so3_log(R):
        tr = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
        theta = np.arccos(tr)
        if theta < 1e-6:
            return np.zeros(3)
        w_hat = (R - R.T) / (2.0 * np.sin(theta))
        return theta * np.array([w_hat[2,1] - 0*w_hat[1,2],
                                 w_hat[0,2] - 0*w_hat[2,0],
                                 w_hat[1,0] - 0*w_hat[0,1]])

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

    def _skew(self, v):
        vx, vy, vz = float(v[0]), float(v[1]), float(v[2])
        return np.array([[ 0.0, -vz,  vy],
                        [ vz,  0.0, -vx],
                        [-vy,  vx,  0.0]], dtype=float)

    def _cube_drop_row_gripper_site(self):
        h, H_xy = self.bf()
        sim = self.env.sim
        qvel_idx = self.env.robots[0].joint_indexes
        ee_site = self.env.robots[0].gripper.important_sites["grip_site"]

        jacp = np.reshape(sim.data.get_site_jacp(ee_site), (3, -1))
        jacr = np.reshape(sim.data.get_site_jacr(ee_site), (3, -1))
        J_full = np.vstack((jacp, jacr))
        Jp = J_full[:3, qvel_idx]                                  
        Jr = J_full[3:, qvel_idx]                                  

        p_site = np.array(sim.data.get_site_xpos(ee_site), dtype=float).reshape(3)

        contact_points = []
        dists_to_site = []

        def is_gripper(name): return ("gripper" in name) or ("robot0" in name and "finger" in name)
        def is_cube(name): return ("cube" in name)

        for i in range(sim.data.ncon):
            c = sim.data.contact[i]
            g1 = sim.model.geom_id2name(c.geom1)
            g2 = sim.model.geom_id2name(c.geom2)
            # Only consider gripper–cube pairs
            if (is_cube(g1) and is_gripper(g2)) or (is_cube(g2) and is_gripper(g1)):
                p_c = np.array(c.pos, dtype=float).reshape(3)  # world-frame contact position
                contact_points.append(p_c)
                dists_to_site.append(np.linalg.norm(p_c - p_site))

        if len(contact_points) > 0:
            idx = int(np.argmin(dists_to_site))
            p_contact = contact_points[idx]
        else:
            p_contact = p_site

        r_world = (p_contact - p_site).reshape(3)
        S_r = self._skew(r_world)

        J_eff = Jp - S_r @ Jr
        J_eff_xy = J_eff[:2, :]

        a = (H_xy.reshape(1, 2) @ J_eff_xy).ravel()
        b = -float(self.cfg["alpha"]) * float(h)
        return a.astype(float), float(b)

    def _wrist_rotation_limit_rows(self):
        sim = self.env.sim
        qvel_idx = self.env.robots[0].joint_indexes
        omega_max = float(self.cfg.get("omega_max", 0.2))
        limit_axes = self.cfg.get("limit_axes", "z")

        ee_site = self.env.robots[0].gripper.important_sites["grip_site"]
        Jr_site = np.reshape(sim.data.get_site_jacr(ee_site), (3, -1))
        Jr = Jr_site[:, qvel_idx]

        axes = [2] if limit_axes == "z" else [0, 1, 2]
        As, Bs = [], []
        for ax in axes:
            row = Jr[ax, :].astype(float)
            As.append(+row)
            Bs.append(-omega_max)
            As.append(-row)
            Bs.append(-omega_max)
        return As, Bs

    def objective_terms(self, env):
        sim = self.env.sim
        qvel_idx = self.env.robots[0].joint_indexes
        ee_site = self.env.robots[0].gripper.important_sites["grip_site"]

        if self._R0 is None:
            self._R0 = self._mat_from_site(sim, ee_site)

        R = self._mat_from_site(sim, ee_site)
        R0 = self._R0
        e_rot = self._so3_log(R0.T @ R)

        kp = float(self.cfg.get("ori_kp", 2.0))
        omega_ref_full = -kp * e_rot

        axes = [2] if self.cfg.get("ori_axes", "z") == "z" else [0, 1, 2]
        W_diag = np.zeros(3)
        for ax in axes:
            W_diag[ax] = float(self.cfg.get("ori_hold_weight", 10.0))

        Jr_site = np.reshape(sim.data.get_site_jacr(ee_site), (3, -1))
        Jr = Jr_site[:, qvel_idx]

        W2 = np.diag(W_diag**2)
        P_extra = Jr.T @ W2 @ Jr
        q_extra = -(Jr.T @ (W2 @ omega_ref_full))

        return P_extra, q_extra

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


class CollisionQPFilter:
    def __init__(self, env, cbf_modules):
        self.env = env
        self.modules = cbf_modules

    @staticmethod
    def _solve_qp(u_des, a_list, b_list, P_extra=None, q_extra=None,
                  *, drop_tol=1e-10, normalize_rows=True):
        u_des = np.asarray(u_des, dtype=float).reshape(-1)
        n = u_des.size

        A_clean, b_clean = [], []
        for a, b in zip(a_list, b_list):
            a = np.asarray(a, dtype=float).reshape(n)
            b = float(b)
            if not np.all(np.isfinite(a)) or not np.isfinite(b):
                continue
            norm = np.linalg.norm(a)
            if norm < drop_tol:
                continue
            if normalize_rows:
                a = a / norm
                b = b / norm
            A_clean.append(a)
            b_clean.append(b)

        m = len(A_clean)
        P = sp.eye(n, format="csc")
        q = -u_des.copy()

        if P_extra is not None and q_extra is not None:
            P = (P + sp.csc_matrix(P_extra)) if not isinstance(P_extra, sp.csc_matrix) else (P + P_extra)
            q = (q + q_extra)

        if m == 0:
            if P_extra is None:
                u = u_des
            else:
                M = (sp.eye(n, format="csc") + sp.csc_matrix(P_extra)).tocsc()
                rhs = u_des - q_extra
                u = np.linalg.solve(M.toarray(), rhs)
            return u, np.zeros(0, dtype=float), True

        A_mat = np.stack(A_clean, axis=0)                  
        b_arr = np.array(b_clean, dtype=float)             

        Au = sp.csc_matrix(A_mat)
        l = b_arr
        uvec = np.full(m, np.inf, dtype=float)

        prob = osqp.OSQP()
        prob.setup(P=P, q=q, A=Au, l=l, u=uvec,
                   verbose=False, polish=False, eps_abs=1e-6, eps_rel=1e-6)
        res = prob.solve()

        ok = (res.info.status_val in (1, 2)) and (res.x is not None)
        if not ok:
            return u_des, np.zeros(m, dtype=float), False

        x = np.array(res.x, dtype=float)
        y = np.array(res.y, dtype=float) if res.y is not None else np.zeros(m, dtype=float)
        return x, y, True

    def apply(self, u_des):
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

        P_extra_total = None
        q_extra_total = None
        for mod in self.modules:
            if hasattr(mod, "objective_terms"):
                obj = mod.objective_terms(self.env)
                if obj is not None:
                    P_e, q_e = obj
                    P_extra_total = (P_e if P_extra_total is None else P_extra_total + P_e)
                    q_extra_total = (q_e if q_extra_total is None else q_extra_total + q_e)

        u_safe, y, ok = self._solve_qp(u_nom, all_As, all_Bs,
                                       P_extra=P_extra_total, q_extra=q_extra_total)

        u_act[:7] = u_safe if ok else u_nom

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
