import osqp
import numpy as np
import scipy.sparse as sp

class TableCollisionFilter:
    def __init__(self, env, alpha=10, margin=0.00, max_step=None):
        self.env = env
        self.alpha = float(alpha)
        self.margin = float(margin)
        self.max_step = None if max_step is None else np.array(max_step, dtype=float).reshape(-1)
        if self.max_step is not None and self.max_step.size == 1:
            self.max_step = np.repeat(self.max_step.item(), 3)

        self.control_freq = float(getattr(env, "control_freq", 20.0))
        self.dt = 1.0 / self.control_freq

        self.h = 0.0
        self.grad = np.zeros(3)
        self.pos = np.zeros(3)

    def superquadric(self):
        e1, e2 = 0.15, 0.15

        kappa = 1e-12

        cx = 0.0
        cy = 0.0
        cz = float(self.env.model.mujoco_arena.table_offset[2])

        Lx = float(self.env.table_full_size[0]) / 2.0
        Ly = float(self.env.table_full_size[1]) / 2.0
        Lz = float(self.env.table_full_size[2]) / 2.0

        p = self.env.sim.data.site_xpos[self.env.robots[0].eef_site_id][:3]
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

        F = Axy + Sz
        self.h = F - (1.0 + self.margin)

        dSx_dx = (2.0 / e2) * (ax ** (2.0 / e2 - 1.0)) * sgnx * (1.0 / Lx)
        dSy_dy = (2.0 / e2) * (ay ** (2.0 / e2 - 1.0)) * sgny * (1.0 / Ly)

        sum_S = Sx + Sy

        if sum_S > 0.0:
            common_xy = (e2 / e1) * (sum_S ** (e2 / e1 - 1.0))
        else:
            common_xy = 0.0

        dA_dx = common_xy * dSx_dx
        dA_dy = common_xy * dSy_dy

        dSz_dz = (2.0 / e1) * (az ** (2.0 / e1 - 1.0)) * sgnz * (1.0 / Lz)

        self.grad = np.array([dA_dx, dA_dy, dSz_dz], dtype=float)

    def hdot(self):
        self.superquadric()
        h = float(self.h)
        H_sem = self.grad.reshape(1, 3)

        ee_site_name = self.env.robots[0].gripper.important_sites["grip_site"]
        jacp = self.env.sim.data.get_site_jacp(ee_site_name) 
        jacr = self.env.sim.data.get_site_jacr(ee_site_name)
        jacp = np.reshape(jacp, (3, -1))
        jacr = np.reshape(jacr, (3, -1))
        J_full = np.vstack([jacp, jacr]) 

        qvel_idx = self.env.robots[0].joint_indexes
        J_robot = J_full[:, qvel_idx]
        J_robot_pos = J_robot[0:3, :]

        a = (H_sem @ J_robot_pos).ravel()

        return a, h


    def enforce_cbf_osqp(self, u_des, a, h, alpha = None, umin = None, umax = None, rho = 1e5):
        if alpha is None:
            alpha = self.alpha

        a = np.asarray(a).reshape(-1)
        u_des = np.asarray(u_des).reshape(-1)
        n= u_des.size
        b = float(-alpha * h)

        P = sp.block_diag([sp.eye(n), sp.csr_matrix([[rho]])], format="csc")
        q = np.zeros(n+1)
        q[:n] = -u_des

        rows = 2
        A_rows = []
        A_data = []
        A_cols = []

        for j in range(n):
            A_rows.append(0); A_cols.append(j); A_data.append(a[j])
        A_rows.append(0); A_cols.append(n); A_data.append(1.0)

        A_rows.append(1); A_cols.append(n); A_data.append(1.0)

        l = [b, 0.0]
        u = [np.inf, np.inf]

        if umin is not None and umax is not None:
            umin = np.asarray(umin).reshape(n)
            umax = np.asarray(umax).reshape(n)
            for i in range(n):
                rows += 1
                A_rows.append(rows-1); A_cols.append(i); A_data.append(1.0)
                l.append(umin[i]); u.append(np.inf)
            for i in range(n):
                rows += 1
                A_rows.append(rows-1); A_cols.append(i); A_data.append(1.0)
                l.append(-np.inf); u.append(umax[i])

        A = sp.csc_matrix((A_data, (A_rows, A_cols)), shape=(rows, n + 1))
        l = np.asarray(l); u = np.asarray(u)

        prob = osqp.OSQP()
        prob.setup(P=P, q=q, A=A, l=l, u=u, verbose=False, polish=False,
                eps_abs=1e-6, eps_rel=1e-6)
        res = prob.solve()

        if res.info.status_val not in (1, 2):
            return u_des, 0.0

        x = res.x
        u_safe = x[:n]
        delta = float(x[n])
        return u_safe, delta

    def apply(self, u_des):
        u_act = u_des.copy()
        u_nom = u_act[:7]
        a,h = self.hdot()
        u_safe, _ = self.enforce_cbf_osqp(u_des=u_nom,a=a,h=h,alpha=self.alpha)
        u_act[:7] = u_safe
        return u_act