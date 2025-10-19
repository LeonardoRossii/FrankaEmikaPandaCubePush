import osqp
import numpy as np
import scipy.sparse as sp

class TableCollisionFilter:
    def __init__(self, env, alpha=10, margin=0.0):
        self.env = env
        self.alpha = alpha
        self.margin = margin
        self.h = 0.0
        self.grad = np.zeros(3)
        self.pos = np.zeros(3)

    def superquadric(self, p):
        e1, e2 = 1.0, 1.0
        kappa = 1e-12

        cx = 0.0
        cy = 0.0
        cz = self.env.model.mujoco_arena.table_offset[2]

        Lx = float(self.env.table_full_size[0]) / 2.0
        Ly = float(self.env.table_full_size[1]) / 2.0
        Lz = float(self.env.table_full_size[2]) / 2.0

        x, y, z = float(p[0]), float(p[1]), float(p[2])
        self.pos = np.array([x, y, z], dtype=float)

        ux = (x - cx) / Lx
        uy = (y - cy) / Ly
        uz = (z - cz) / Lz

        ax = np.sqrt(ux * ux + kappa)
        ay = np.sqrt(uy * uy + kappa)
        az = np.sqrt(uz * uz + kappa)

        sgnx = ux / ax
        sgny = uy / ay
        sgnz = uz / az

        Sx = (ax) ** (2.0 / e2)
        Sy = (ay) ** (2.0 / e2)
        Axy = (Sx + Sy) ** (e2 / e1) 
        Sz = (az) ** (2.0 / e1)      
        sum_S = Sx + Sy
        F = Axy + Sz

        self.h = F - (1.0 + self.margin)

        dSx_dx = (2.0 / e2) * (ax ** (2.0 / e2 - 1.0)) * sgnx * (1.0 / Lx)
        dSy_dy = (2.0 / e2) * (ay ** (2.0 / e2 - 1.0)) * sgny * (1.0 / Ly)
        common_xy = (e2 / e1) * (sum_S ** (e2 / e1 - 1.0)) 
        dA_dx = common_xy * dSx_dx
        dA_dy = common_xy * dSy_dy
        dSz_dz = (2.0 / e1) * (az ** (2.0 / e1 - 1.0)) * sgnz * (1.0 / Lz)

        self.grad = np.array([dA_dx, dA_dy, dSz_dz], dtype=float)
        return self.h, self.grad

    def bodies(self):
        all_bodies = [self.env.sim.model.body_id2name(i)for i in range(self.env.sim.model.nbody)]
        #print(all_bodies)
        bodies = ["gripper0_eef",
                  "robot0_link7",
                  "robot0_link6",
                  "robot0_link5",
                  "robot0_link4"
                  ]
        
        n = len(bodies)
        
        xs = [None]*n
        hs = [None]*n
        Hs = [None]*n
        Js = [None]*n
        As = [None]*n

        for i in range(n):
            xs[i] = self.env.sim.data.body_xpos[self.env.sim.model.body_name2id(bodies[i])][:3]
            hs[i], Hs[i] = self.superquadric(xs[i])

            jacp = self.env.sim.data.get_body_jacp(bodies[i]) 
            jacr = self.env.sim.data.get_body_jacr(bodies[i])
            jacp = np.reshape(jacp, (3, -1))
            jacr = np.reshape(jacr, (3, -1))
            J_full = np.vstack([jacp, jacr]) 
            qvel_idx = self.env.robots[0].joint_indexes
            J_robot = J_full[:, qvel_idx]
            J_robot_pos = J_robot[0:3, :]
            Js[i] = J_robot_pos
            gmax = 5.0  # tune
            Js[i] = np.clip(Js[i], -gmax, gmax)
            As[i] = (Hs[i] @ Js[i]).ravel()
        
        print("A[gripper]", As[0])
        print("A[link7]", As[1])


        return As, hs


    def enforce_cbf_osqp(self, u_des, a_list, h_list, rho = 1e5,  u_min=None, u_max=None, slack_mode= "per"):



        u_des = np.asarray(u_des).reshape(-1) # Force u_eds to 1-D array
        n = u_des.size

        a_arr = [np.asarray(ai, dtype=float).reshape(n) for ai in a_list]
        h_arr = [float(hi) for hi in h_list]
        m = len(a_arr)
        assert len(h_arr) == m


        b_arr = np.array([-self.alpha * hi for hi in h_arr], dtype=float)

        if slack_mode == "per":
            nd = m
        elif slack_mode == "shared":
            nd = 1
        else:
            raise ValueError("slack_mode must be 'per' or 'shared'")

        eps = 1e-9

        Pu = sp.eye(n, format="csc") * (1.0 + eps)

        if slack_mode == "per":
            if np.isscalar(rho):
                rho_vec = np.full(nd, float(rho))
            else:
                rho_vec = np.asarray(rho, dtype=float).reshape(nd)
            Pd = sp.diags(rho_vec + eps, format="csc")
        else:
            Pd = sp.csr_matrix([[float(rho) + eps]])

        P = sp.block_diag([Pu, Pd], format="csc")
        q = np.zeros(n + nd)
        q[:n] = -u_des

        A_blocks = []
        l_list, u_list = [], []

        # 1) CBF constraints: a_i^T u + delta_i >= b_i
        # For shared slack, delta is the same column for all rows.
        if slack_mode == "per":
            # Build [A_u | I_m] where A_u has rows a_i^T
            Au = sp.csr_matrix(np.vstack([ai for ai in a_arr]))  # (m × n)
            Id = sp.eye(m, format="csc")                         # (m × m)
            A_cbf = sp.hstack([Au, Id], format="csc")            # (m × (n+m))
        else:
            Au = sp.csr_matrix(np.vstack([ai for ai in a_arr]))  # (m × n)
            ones = sp.csr_matrix(np.ones((m, 1)))                # (m × 1)
            A_cbf = sp.hstack([Au, ones], format="csc")          # (m × (n+1))

        A_blocks.append(A_cbf)
        l_list.extend(b_arr.tolist())
        u_list.extend([np.inf]*m)

        # 2) Slack nonnegativity: delta >= 0
        if slack_mode == "per":
            A_s = sp.hstack([sp.csr_matrix((m, n)), sp.eye(m, format="csc")], format="csc")
            A_blocks.append(A_s)
            l_list.extend([0.0]*m)
            u_list.extend([np.inf]*m)
        else:
            A_s = sp.hstack([sp.csr_matrix((1, n)), sp.csr_matrix([[1.0]])], format="csc")
            A_blocks.append(A_s)
            l_list.append(0.0)
            u_list.append(np.inf)

        # 3) Box constraints on u (optional but recommended)
        if u_min is None:
            u_min = getattr(self, "u_min", None)
        if u_max is None:
            u_max = getattr(self, "u_max", None)

        if u_min is not None and u_max is not None:
            u_min = np.asarray(u_min, dtype=float).reshape(n)
            u_max = np.asarray(u_max, dtype=float).reshape(n)
            Iu = sp.eye(n, format="csc")

            #  u <= u_max
            A_ub = sp.hstack([Iu, sp.csr_matrix((n, nd))], format="csc")
            A_blocks.append(A_ub)
            l_list.extend([-np.inf]*n)
            u_list.extend(u_max.tolist())

            #  -u <= -u_min  (i.e., u >= u_min)
            A_lb = sp.hstack([-Iu, sp.csr_matrix((n, nd))], format="csc")
            A_blocks.append(A_lb)
            l_list.extend([-np.inf]*n)
            u_list.extend((-u_min).tolist())

        # Stack
        A = sp.vstack(A_blocks, format="csc")
        l = np.asarray(l_list, dtype=float)
        u = np.asarray(u_list, dtype=float)

        # --- solve ---
        prob = osqp.OSQP()
        prob.setup(P=P, q=q, A=A, l=l, u=u, verbose=False, polish=True, eps_abs=1e-6, eps_rel=1e-6)
        res = prob.solve()

        if res.info.status_val not in (1, 2):
            # Fallback: clamp u_des to bounds if present, else return u_des
            if u_min is not None and u_max is not None:
                u_fallback = np.clip(u_des, u_min, u_max)
            else:
                u_fallback = u_des
            return u_fallback, (np.zeros(nd) if slack_mode=="per" else 0.0)

        x = res.x
        u_safe = x[:n]
        delta = x[n:] if slack_mode == "per" else float(x[n])
        return u_safe, delta

    def apply(self, u_des):
        u_act = u_des.copy()
        u_nom = u_act[:7]
        a_list, h_list = self.bodies()
        u_safe, _ = self.enforce_cbf_osqp(u_des=u_nom,a_list=a_list,h_list=h_list)
        u_act[:7] = u_safe
        return u_act