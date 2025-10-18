import osqp
import numpy as np
import scipy.sparse as sp

class CubeDropFilter():
    def __init__(self, env, alpha = 1.0, margin =0.0):
        self.env = env
        self.alpha = alpha
        self.margin = margin
        self.h = None
        self.H = None

    def superquadric(self):
        cx = 0.0
        cy = 0.0

        Lx = self.env.table_full_size[0] / 2.0
        Ly = self.env.table_full_size[1] / 2.0

        rx = 0.015
        ry = 0.015

        mx = Lx - rx
        my = Ly - ry
     
        p = self.env.sim.data.body_xpos[self.env.cube_body_id]
        xc = p[0]
        yc = p[1]

        e1 = 0.5
        e2 = 0.5

        kappa = 1e-8
        ux = (xc-cx)/mx
        uy = (yc-cy)/my

        ax = np.sqrt(ux*ux + kappa)
        ay = np.sqrt(uy*uy + kappa)
        
        Sx = ax**(2.0/e2)
        Sy = ay**(2.0/e2)
        S = Sx+Sy
        F = S**(e2/e1)
        h = 1.0 - F - self.margin
        self.h = h

        sgnx = ux/ax
        sgny = uy/ay

        dSx_dxc = (2.0/e2) * (ax**(2.0/e2 - 1.0)) * sgnx * (1.0/mx)
        dSy_dyc = (2.0/e2) * (ay**(2.0/e2 - 1.0)) * sgny * (1.0/my)
        
        c   = (e2/e1) * (S**(e2/e1 - 1.0)) if S>0 else 0.0
        dh_dxc = - c * dSx_dxc
        dh_dyc = - c * dSy_dyc
        H = np.array([[dh_dxc, dh_dyc]])
        self.H = H

    def contact(self):
        cubeName = "cube_g0"
        tableName = "table"
        for contact in self.env.sim.data.contact:
            geom1 = self.env.sim.model.geom_id2name(contact.geom1)
            geom2 = self.env.sim.model.geom_id2name(contact.geom2)
            if ((cubeName in geom1) or (cubeName in geom2)) and ((tableName not in geom1) and (tableName not in geom2)):
                if cubeName in geom1: return geom2
                else: return geom1

    def hdot(self):
        self.superquadric()
        h = float(self.h)
        H_sem = self.H.reshape(1, 2)

        part = None
        print(part)

        """all_sites = [self.env.sim.model.site_id2name(i) for i in range(self.env.sim.model.nsite)]
        print(all_sites)"""

        """all_geoms = [self.env.sim.model.geom_id2name(i) for i in range(self.env.sim.model.ngeom)]
        print(all_geoms)"""

        """all_bodies = [self.env.sim.model.body_id2name(i)for i in range(self.env.sim.model.nbody)]
        print(all_bodies)"""

        if part is None:
            ee_site_name = self.env.robots[0].gripper.important_sites["grip_site"]
            jacp = self.env.sim.data.get_site_jacp(ee_site_name) 
            jacr = self.env.sim.data.get_site_jacr(ee_site_name)
        else:
            jacp = self.env.sim.data.get_geom_jacp(part)
            jacr = self.env.sim.data.get_geom_jacr(part)

        jacp = np.reshape(jacp, (3, -1))
        jacr = np.reshape(jacr, (3, -1))
        J_full = np.vstack([jacp, jacr])

        qvel_idx = self.env.robots[0].joint_indexes
        J_robot = J_full[:, qvel_idx]
        J_robot_pos = J_robot[0:3, :]
        J_cube_xy = J_robot_pos[:2, :]

        a = (H_sem @ J_cube_xy).ravel()

        return a, h
    
    def enforce_cbf_osqp(self, u_des, a, h, rho = 1e5):

        u_des = np.asarray(u_des).reshape(-1)
        n = u_des.size

        a = np.asarray(a).reshape(-1)
        b = float(-self.alpha * h)
        
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
        u_safe, _ = self.enforce_cbf_osqp(u_des=u_nom,a=a,h=h)
        u_act[:7] = u_safe
        return u_act







