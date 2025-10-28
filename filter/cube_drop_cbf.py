import numpy as np
from .cbf_base import CBFModule

class CubeDropCBF(CBFModule):
    def __init__(self, env):
        self.env = env
        self.cfg = {
            "alpha": 0.1,   # Linear cbf function (need to be of class K)
            "margin": 0.02, # inflate XY safe set by this margin (m)
            "rx": 0.020,    # x-radius deduction from table half-width (m)
            "ry": 0.020,    # y-radius deduction from table half-depth (m)
            "e1": 0.1,      # outer exponent for super-ellipse shaping
            "e2": 0.1,      # inner exponent for super-ellipse shaping
            "kappa": 1e-12, # smoothing term to keep gradients finite near 0
        }

    def bf(self):
        cfg = self.cfg

        # Table center in x y coordinates 
        cx, cy = 0.0, 0.0

        # Table half-sizes and effective radii
        Lx = self.env.table_full_size[0] / 2.0
        Ly = self.env.table_full_size[1] / 2.0
        mx, my = Lx - cfg["rx"], Ly - cfg["ry"]

        # Cube world position (x, y)
        p = self.env.sim.data.body_xpos[self.env.cube_body_id]
        xc, yc = float(p[0]), float(p[1])

        # Super-ellipse parameters and normalized coords
        e1, e2, kappa = cfg["e1"], cfg["e2"], cfg["kappa"]
        ux, uy = (xc - cx) / mx, (yc - cy) / my

        # Approximation of the absolute value term
        # Smooth |u| ≈ sqrt(u^2 + kappa) to keep derivatives finite
        ax, ay = np.sqrt(ux * ux + kappa), np.sqrt(uy * uy + kappa)

        # Inner terms
        Sx = ax ** (2.0 / e2)
        Sy = ay ** (2.0 / e2)
        S = Sx + Sy
        F = S ** (e2 / e1)

        # Build barrier function 
        h = 1.0 - (F - cfg["margin"])

        # Gradients terms computation
        sgnx, sgny = ux / ax, uy / ay
        dSx_dxc = (2.0 / e2) * (ax ** (2.0 / e2 - 1.0)) * sgnx / mx
        dSy_dyc = (2.0 / e2) * (ay ** (2.0 / e2 - 1.0)) * sgny / my
        c = (e2 / e1) * (S ** (e2 / e1 - 1.0)) if S > 0 else 0.0
        dh_dxc = -c * dSx_dxc
        dh_dyc = -c * dSy_dyc
        H_xy = np.array([dh_dxc, dh_dyc], dtype=float)

        return float(h), H_xy

    @staticmethod
    def _skew(v):
        # Return 3x3 skew-symmetric matrix [v]x such that:
        # [v]x w = v x w
        vx, vy, vz = float(v[0]), float(v[1]), float(v[2])
        return np.array([[ 0.0, -vz,  vy],
                         [ vz,  0.0, -vx],
                         [-vy,  vx,  0.0]], dtype=float)

    def _effective_jacobian_at_closest_contact(self):
        sim = self.env.sim
        qvel_idx = self.env.robots[0].joint_indexes
        ee_site = self.env.robots[0].gripper.important_sites["grip_site"]
        jacp = np.reshape(sim.data.get_site_jacp(ee_site), (3, -1))
        jacr = np.reshape(sim.data.get_site_jacr(ee_site), (3, -1))
        J_full = np.vstack((jacp, jacr))
        Jp = J_full[:3, qvel_idx]
        Jr = J_full[3:, qvel_idx]
        p_site = np.array(sim.data.get_site_xpos(ee_site), dtype=float).reshape(3)

        # Find contact point if there is at least one
        contact_points, dists_to_site = [], []
        def is_gripper(name): 
            return ("gripper" in name) or ("robot0" in name and "finger" in name)
        def is_cube(name): 
            return ("cube" in name)
        for i in range(sim.data.ncon):
            c = sim.data.contact[i]
            g1 = sim.model.geom_id2name(c.geom1)
            g2 = sim.model.geom_id2name(c.geom2)
            if (is_cube(g1) and is_gripper(g2)) or (is_cube(g2) and is_gripper(g1)):
                p_c = np.array(c.pos, dtype=float).reshape(3)
                contact_points.append(p_c)
                dists_to_site.append(np.linalg.norm(p_c - p_site))
        p_contact = (
            contact_points[int(np.argmin(dists_to_site))]
            if contact_points else p_site
        )
        #p_contact = self.env.sim.data.body_xpos[self.env.cube_body_id]

        # Shift linear Jacobian to contact:
        # J_eff = Jp - [r]× Jr
        r_world = (p_contact - p_site).reshape(3)
        S_r = self._skew(r_world)
        J_eff = Jp - S_r @ Jr
        return J_eff, p_contact

    def _row_xy_drop(self):
        h, H_xy = self.bf()
        J_eff, _ = self._effective_jacobian_at_closest_contact()
        J_eff_xy = J_eff[:2, :]
        a = (H_xy.reshape(1, 2) @ J_eff_xy).ravel()
        b = -float(self.cfg["alpha"]) * float(h)
        return a.astype(float), float(b)

    def _row_z_no_upward_velocity(self):
        J_eff, _ = self._effective_jacobian_at_closest_contact()
        a = (-J_eff[2, :]).ravel().astype(float)
        b = 0.0
        return a, b
    
    def constraints(self, env):
        self.env = env
        As, bs = [], []
        try:
            a_xy, b_xy = self._row_xy_drop()
            As.append(a_xy)
            bs.append(b_xy)
            a_z, b_z = self._row_z_no_upward_velocity()
            As.append(a_z)
            bs.append(b_z)
        except Exception as e:
            return [], []
        return As, bs
    
    def objective_terms(self, env):
        return None
