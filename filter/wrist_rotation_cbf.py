import numpy as np
from .cbf_base import CBFModule

class WristRotationCBF(CBFModule):
    """
    Constrains wrist angular velocity (CBF-style linear rows) and
    adds a quadratic term to softly hold orientation.
    """
    def __init__(self, env):
        self.env = env
        self.cfg = {
            "omega_max": 0.5,       # angular velocity cap
            "limit_axes": "all",    # "z" or "all"
            "ori_hold_weight": 10.0,# weight of quadratic term
            "ori_kp": 2.0,          # gain from orientation error to omega_ref
            "ori_axes": "all",      # "z" or "all"
        }
        self._R0 = None

    @staticmethod
    def _mat_from_site(sim, site_name):
        return np.array(sim.data.get_site_xmat(site_name), dtype=float).reshape(3, 3)

    @staticmethod
    def _so3_log(R):
        tr = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
        theta = np.arccos(tr)
        if theta < 1e-6:
            return np.zeros(3)
        w_hat = (R - R.T) / (2.0 * np.sin(theta))
        return theta * np.array([w_hat[2,1], w_hat[0,2], w_hat[1,0]])

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
            As.append(+row); Bs.append(-omega_max)
            As.append(-row); Bs.append(-omega_max)
        return As, Bs

    def constraints(self, env):
        self.env = env
        try:
            return self._wrist_rotation_limit_rows()
        except Exception:
            return [], []

    def objective_terms(self, env):
        sim = self.env.sim
        qvel_idx = self.env.robots[0].joint_indexes
        ee_site = self.env.robots[0].gripper.important_sites["grip_site"]

        if self._R0 is None:
            self._R0 = self._mat_from_site(sim, ee_site)

        R = self._mat_from_site(sim, ee_site)
        e_rot = self._so3_log(self._R0.T @ R)

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
