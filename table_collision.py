import osqp
import numpy as np
import scipy.sparse as sp

class TableCollisionFilter:
    def __init__(self, env):
        self.env = env
        self.alphas = None
        self.cx = 0.0
        self.cy = 0.0
        self.cz = self.env.model.mujoco_arena.table_offset[2]
        print(self.env.model.mujoco_arena.table_offset[2])
        print(self.cz)
        self.c = np.array([self.cx, self.cy, self.cz])
        self.alpha = 3.0
        self.robot_config = {
            "bodies": {
                "gripper0_eef": {"alpha": 10.0, "margin": 0.05, "e1": 0.2, "e2":0.1},
                "gripper0_leftfinger": {"alpha": 10.0, "margin": 0.05, "e1": 0.2, "e2":0.1},
                "gripper0_rightfinger": {"alpha": 10.0, "margin": 0.05, "e1": 0.2, "e2":0.1},
                "robot0_link7": {"alpha": 2, "margin": 0.05, "e1": 0.2, "e2":0.2},
                "robot0_link6": {"alpha": 2, "margin": 0.05, "e1": 0.2, "e2":0.2},
                "robot0_link5": {"alpha": 2, "margin": 0.05, "e1": 0.2, "e2":0.2},
                "robot0_link4": {"alpha": 2, "margin": 0.05, "e1": 0.2, "e2":0.2},
                "robot0_link3": {"alpha": 2, "margin": 0.05, "e1": 0.2, "e2":0.2},
            },
            "geoms": {
                "robot0_link7_collision": {"alpha": 2, "margin": 0.05, "e1": 0.2, "e2":0.2},
                "robot0_link6_collision": {"alpha": 2, "margin": 0.05, "e1": 0.2, "e2":0.2},
                "robot0_link5_collision": {"alpha": 2, "margin": 0.05, "e1": 0.2, "e2":0.2},
                "robot0_link4_collision": {"alpha": 2, "margin": 0.05, "e1": 0.2, "e2":0.2},
                "robot0_link3_collision": {"alpha": 2, "margin": 0.05, "e1": 0.2, "e2":0.2},
            }
        }


    def cbf(self, p, margin, e1= 0.1, e2 = 0.1):

        e1 = max(e1, 0.5)

        k = 1e-3

        Lx = self.env.table_full_size[0] / 2.0
        Ly = self.env.table_full_size[1] / 2.0
        Lz = self.env.table_full_size[2] / 2.0

        cx = self.cx
        cy = self.cy
        cz = self.cz

        x = p[0]
        y = p[1]
        z = p[2]

        ux = (x - cx) / Lx
        uy = (y - cy) / Ly
        uz = (z - cz) / Lz

        ax = np.sqrt(ux * ux + k)
        ay = np.sqrt(uy * uy + k)
        az = np.sqrt(uz * uz + k)

        sgnx = ux / ax
        sgny = uy / ay
        sgnz = uz / az

        Sx = ax ** (2.0 / e2)
        Sy = ay ** (2.0 / e2)
        Sxy = Sx + Sy
        Axy = Sxy ** (e2 / e1) 
        Sz = az ** (2.0 / e1)      
        
        F = Axy + Sz
        h = F - (1.0 + margin)

        dSx_dx = (2.0 / e2) * (ax ** (2.0 / e2 - 1.0)) * sgnx * (1.0 / Lx)
        dSy_dy = (2.0 / e2) * (ay ** (2.0 / e2 - 1.0)) * sgny * (1.0 / Ly)
        dA_dx = (e2 / e1) * (Sxy ** (e2 / e1 - 1.0))  * dSx_dx
        dA_dy = (e2 / e1) * (Sxy ** (e2 / e1 - 1.0))  * dSy_dy
        dSz_dz = (2.0 / e1) * (az ** (2.0 / e1 - 1.0)) * sgnz * (1.0 / Lz)

        H = np.array([dA_dx, dA_dy, dSz_dz])

        return h, H

    def constraints(self):

        As, bs = [], []
        sim = self.env.sim
        model = sim.model
        robot_joint_idx = self.env.robots[0].joint_indexes

        for body_name, params in self.robot_config["bodies"].items():
            alpha = params["alpha"]
            margin = params["margin"]

            x = sim.data.body_xpos[model.body_name2id(body_name)][:3]
            if np.linalg.norm(x[2] - self.cz) < 10.0:
                h, H = self.cbf(x, margin, e1=params["e1"], e2=params["e2"])
                bs.append(-alpha * h)
                """print("Body name", body_name)
                print("H", H)
                print("\n\n")"""
                jacp = np.reshape(sim.data.get_body_jacp(body_name), (3, -1))
                jacr = np.reshape(sim.data.get_body_jacr(body_name), (3, -1))
                J_full = np.vstack((jacp, jacr))

                J_robot_pos = J_full[:3, robot_joint_idx]
                a = (H @ J_robot_pos).ravel()
                As.append(a)

        for geom_name, params in self.robot_config["geoms"].items():
            alpha = params["alpha"]
            margin = params["margin"]

            x = sim.data.geom_xpos[model.geom_name2id(geom_name)][:3]
            if np.linalg.norm(x[2] - self.cz) < 10.0:
                h, H = self.cbf(x, margin, e1=params["e1"], e2=params["e2"])
                bs.append(-alpha * h)

                jacp = np.reshape(sim.data.get_geom_jacp(geom_name), (3, -1))
                jacr = np.reshape(sim.data.get_geom_jacr(geom_name), (3, -1))
                J_full = np.vstack((jacp, jacr))

                J_robot_pos = J_full[:3, robot_joint_idx]
                a = (H @ J_robot_pos).ravel()
                As.append(a)
        return As, bs



    def solve(self, u_des, a_list, b_list, rho=1e5):
        if len(a_list)>0:
        
            u_des = np.asarray(u_des, dtype=float).reshape(-1)
            n = u_des.size

            a_arr = [np.asarray(ai, dtype=float).reshape(n) for ai in a_list]
            b_arr = [float(hi) for hi in b_list]
            m = len(a_arr)

            eps = 1e-9
            Pu = sp.eye(n, format="csc") * (1.0 + eps)          
            Pd = sp.csr_matrix([[float(rho) + eps]])            
            P = sp.block_diag([Pu, Pd], format="csc")

            q = np.zeros(n + 1)
            q[:n] = -u_des

            Au = sp.csr_matrix(np.vstack(a_arr))                
            ones = sp.csr_matrix(np.ones((m, 1)))               
            A_cbf = sp.hstack([Au, ones], format="csc")         
            l_cbf = b_arr                                       
            u_cbf = np.full(m, np.inf)                         

            A_s = sp.hstack([sp.csr_matrix((1, n)), sp.csr_matrix([[1.0]])], format="csc")
            l_s = np.array([0.0])
            u_s = np.array([np.inf])

            A = sp.vstack([A_cbf, A_s], format="csc")
            l = np.concatenate([l_cbf, l_s])
            u = np.concatenate([u_cbf, u_s])

            prob = osqp.OSQP()
            prob.setup(P=P, q=q, A=A, l=l, u=u,
                    verbose=False, polish=True, eps_abs=1e-6, eps_rel=1e-6)
            res = prob.solve()

            if res.info.status_val not in (1, 2):
                return u_des, 0.0

            x = res.x
            u_safe = x[:n]
            delta = float(x[n])
            return u_safe, delta
        else:
            return u_des, None

    def apply(self, u_des):
        u_act = u_des.copy()
        u_nom = u_act[:7]
        a_list, h_list = self.constraints()
        u_safe, _ = self.solve(u_nom, a_list, h_list)
        u_act[:7] = u_safe
        return u_act