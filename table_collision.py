import osqp
import numpy as np
import scipy.sparse as sp

class CBFModule:
    def constraints(self,_):
        raise NotImplementedError

class TableTopCBF(CBFModule):
    def __init__(self, env):
        self.env = env
        self.cz = self.env.model.mujoco_arena.table_offset[2]
        self.robot_config ={
            "bodies": {
                "gripper0_eef":        {"alpha": 2.0, "margin": 0.01},
                "gripper0_leftfinger": {"alpha": 2.0, "margin": 0.01},
                "gripper0_rightfinger":{"alpha": 2.0, "margin": 0.01},
                "robot0_link7":        {"alpha": 0.5, "margin": 0.05},
                "robot0_link6":        {"alpha": 0.5, "margin": 0.05},
                "robot0_link5":        {"alpha": 0.5, "margin": 0.05},
                "robot0_link4":        {"alpha": 0.5, "margin": 0.05},
                "robot0_link3":        {"alpha": 0.5, "margin": 0.05},
            },
            "geoms": {
                "robot0_link7_collision": {"alpha": 0.5, "margin": 0.05},
                "robot0_link6_collision": {"alpha": 0.5, "margin": 0.05},
                "robot0_link5_collision": {"alpha": 0.5, "margin": 0.05},
                "robot0_link4_collision": {"alpha": 0.5, "margin": 0.05},
                "robot0_link3_collision": {"alpha": 0.5, "margin": 0.05},
            }
        }

    @staticmethod
    def _table_plane_cbf(env, p, margin):
        z_top = float(env.model.mujoco_arena.table_offset[2])
        h = float(p[2]) - (z_top + margin)
        H = np.array([0.0, 0.0, 1.0])
        return h, H

    def constraints(self, env):
        As, bs = [], []
        sim = env.sim
        model = sim.model
        robot_joint_idx = env.robots[0].joint_indexes
        print(robot_joint_idx)

        for body_name, params in self.robot_config["bodies"].items():
            alpha = float(params["alpha"]); margin = float(params["margin"])
            x = sim.data.body_xpos[model.body_name2id(body_name)][:3]
            if np.linalg.norm(x[2] - self.cz) < 10.0:
                h, H = self._table_plane_cbf(env, x, margin)
                jacp = np.reshape(sim.data.get_body_jacp(body_name), (3, -1))
                jacr = np.reshape(sim.data.get_body_jacr(body_name), (3, -1))
                J_full = np.vstack((jacp, jacr))
                J_robot_pos = J_full[:3, robot_joint_idx]
                a = (H @ J_robot_pos).ravel()
                As.append(a)
                bs.append(-alpha * h)

        for geom_name, params in self.robot_config["geoms"].items():
            alpha = float(params["alpha"]); margin = float(params["margin"])
            x = sim.data.geom_xpos[model.geom_name2id(geom_name)][:3]
            if np.linalg.norm(x[2] - self.cz) < 10.0:
                h, H = self._table_plane_cbf(env, x, margin)
                jacp = np.reshape(sim.data.get_geom_jacp(geom_name), (3, -1))
                jacr = np.reshape(sim.data.get_geom_jacr(geom_name), (3, -1))
                J_full = np.vstack((jacp, jacr))
                J_robot_pos = J_full[:3, robot_joint_idx]
                a = (H @ J_robot_pos).ravel()
                As.append(a)
                bs.append(-alpha * h)
        return As, bs

class CubeDropCBF(CBFModule):
    def __init__(self, env):
        self.env = env
        self.cfg = {
            "alpha": 0.25,
            "margin": 0.03,
            "rx": 0.3,
            "ry": 0.0,
            "e1": 0.1,
            "e2": 0.1,
            "kappa": 1e-8,
        }

    def bf(self):
        cfg = self.cfg
        cx, cy = 0.0, 0.0
        Lx = self.env.table_full_size[0] / 2.0
        Ly = self.env.table_full_size[1] / 2.0
        mx, my = Lx - cfg["rx"], Ly - cfg["ry"]
        p = self.env.sim.data.body_xpos[self.env.cube_body_id]
        xc, yc = p[0], p[1]
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
        H_xy = np.array([dh_dxc, dh_dyc])
        return h, H_xy

    def _cube_drop_row_eef(self):
        h, H_xy = self.bf()
        sim = self.env.sim
        ee_site = self.env.robots[0].gripper.important_sites["grip_site"]
        jacp = sim.data.get_site_jacp(ee_site)
        jacr = sim.data.get_site_jacr(ee_site)
        jacp = np.reshape(jacp, (3, -1))
        jacr = np.reshape(jacr, (3, -1))
        J_full = np.vstack([jacp, jacr])
        qvel_idx = self.env.robots[0].joint_indexes
        J_robot = J_full[:, qvel_idx]
        J_pos = J_robot[:3, :]       
        J_xy = J_pos[:2, :]
        a = (H_xy.reshape(1, 2) @ J_xy).ravel()
        alpha = float(self.cfg["alpha"])
        b = -alpha * h
        return a, b

    def constraints(self, env):
        self.env = env
        As, bs = [], []
        try:
            a_cd, b_cd = self._cube_drop_row_eef()
            As.append(a_cd)
            bs.append(b_cd)
        except Exception:
            pass
        return As, bs

class CollisionQPFilter:
    def __init__(self, env, cbf_modules):
        self.env = env
        self.modules = cbf_modules

    @staticmethod
    def _solve_qp(u_des, a_list, b_list):
        u_des = np.asarray(u_des, dtype=float).reshape(-1)
        n = u_des.size

        A_arr = [np.asarray(ai, dtype=float).reshape(n) for ai in a_list]
        b_arr = np.array([float(bi) for bi in b_list], dtype=float)
        m = len(A_arr)
        A_mat = np.stack(A_arr, axis=0)
        Au = sp.csc_matrix(A_mat)   
        l = b_arr                             
        u = np.full(m, np.inf)                
        P = sp.eye(n, format="csc")
        q = -u_des

        prob = osqp.OSQP()
        prob.setup(P=P, q=q, A=Au, l=l, u=u,
                   verbose=False, polish=False, eps_abs=1e-6, eps_rel=1e-6)
        res = prob.solve()
        status_ok = res.info.status_val in (1, 2)
        if not status_ok or res.x is None:
            return u_des, False
        return res.x, res.y, True

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

        u_safe, y, ok = self._solve_qp(u_nom, all_As, all_Bs)
        u_act[:7] = u_safe if ok else u_nom

        efforts = {}
        if ok and len(all_As) > 0:
            A_mat = np.stack([np.asarray(ai, dtype=float).reshape(-1) for ai in all_As], axis=0)  # (m, n)
            for name, lo, hi in groups:
                A_g = A_mat[lo:hi, :]
                y_g = y[lo:hi]
                delta_g = -A_g.T @ y_g  
                efforts[name] = {
                    "delta": delta_g,                 
                    "effort_l2": float(np.linalg.norm(delta_g)),
                    "active_constraints": int(np.count_nonzero(y_g)),
                }
        return u_act, efforts