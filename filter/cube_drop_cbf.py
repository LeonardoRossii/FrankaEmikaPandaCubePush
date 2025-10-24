import numpy as np
from .cbf_base import CBFModule

class CubeDropCBF(CBFModule):
    def __init__(self, env):
        self.env = env
        self.cfg = {
            "alpha": 0.25,
            "margin": 0.020,
            "rx": 0.020,
            "ry": 0.020,
            "e1": 0.1,
            "e2": 0.1,
            "kappa": 1e-12,
        }

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
        H_xy = np.array([dh_dxc, dh_dyc], dtype=float)
        return float(h), H_xy

    @staticmethod
    def _skew(v):
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

        contact_points, dists_to_site = [], []
        def is_gripper(name): return ("gripper" in name) or ("robot0" in name and "finger" in name)
        def is_cube(name): return ("cube" in name)
        for i in range(sim.data.ncon):
            c = sim.data.contact[i]
            g1 = sim.model.geom_id2name(c.geom1)
            g2 = sim.model.geom_id2name(c.geom2)
            if (is_cube(g1) and is_gripper(g2)) or (is_cube(g2) and is_gripper(g1)):
                p_c = np.array(c.pos, dtype=float).reshape(3)
                contact_points.append(p_c)
                dists_to_site.append(np.linalg.norm(p_c - p_site))
        p_contact = contact_points[int(np.argmin(dists_to_site))] if contact_points else p_site

        r_world = (p_contact - p_site).reshape(3)
        S_r = self._skew(r_world)

        J_eff = Jp - S_r @ Jr
        J_eff_xy = J_eff[:2, :]

        a = (H_xy.reshape(1, 2) @ J_eff_xy).ravel()
        b = -float(self.cfg["alpha"]) * float(h)
        return a.astype(float), float(b)

    def constraints(self, env):
        self.env = env
        try:
            a_cd, b_cd = self._cube_drop_row_gripper_site()
            return [a_cd], [b_cd]
        except Exception:
            return [], []

    def objective_terms(self, env):
        return None
