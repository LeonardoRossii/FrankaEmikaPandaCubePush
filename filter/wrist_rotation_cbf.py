import numpy as np
from .cbf_base import CBFModule

class WristRotationCBF(CBFModule):
    def __init__(self, env):
        self.env = env
        self.cfg = {
            "omega_max": 0.5,        # Max allowed wrist angular velocity magnitude (rad/s)
            "limit_axes": "all",     # "z" to limit only yaw, or "all" for roll/pitch/yaw
            "ori_hold_weight": 10.0, # Strength of quadratic penalty that keeps orientation
            "ori_kp": 2.0,           # Gain mapping orientation error → target angular velocity
            "ori_axes": "all",       # Apply orientation hold to "z" only or all axes
        }
        self._R0 = None  # Stored reference orientation

    @staticmethod
    def _mat_from_site(sim, site_name):
        #Return the site's rotation matrix as a 3x3 numpy array
        return np.array(sim.data.get_site_xmat(site_name), dtype=float).reshape(3, 3)

    @staticmethod
    def _so3_log(R):
        """
        Compute the rotational error vector via the SO(3) logarithm map.
        Returns a 3-vector w such that exp(w^) = R.
        If angle is very small, return zero for stability.
        """
        tr = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
        theta = np.arccos(tr)
        if theta < 1e-6:
            return np.zeros(3)
        w_hat = (R - R.T) / (2.0 * np.sin(theta))
        return theta * np.array([w_hat[2,1], w_hat[0,2], w_hat[1,0]])

    def _wrist_rotation_limit_rows(self):
        """
        Construct linear inequality constraints that bound the wrist angular velocity
        For each constrained axis:
            Jr_row @ q̇ ≤  ω_max
           -Jr_row @ q̇ ≤  ω_max
        Which is equivalent to:
            |ω_axis| ≤ ω_max
        """
        sim = self.env.sim

        # Indices of actuated robot joints 
        # Used to extract proper Jacobian columns
        qvel_idx = self.env.robots[0].joint_indexes

        # Get the configuration parameters if are defined
        # If not set them to 0.2 and "z"
        omega_max = self.cfg.get("omega_max", 0.2)
        limit_axes = self.cfg.get("limit_axes", "z")

        # Use the gripper end-effector site
        ee_site = self.env.robots[0].gripper.important_sites["grip_site"]

        # Get the rotational Jacobian
        Jr_site = np.reshape(sim.data.get_site_jacr(ee_site), (3, -1))
        Jr = Jr_site[:, qvel_idx]

        # Determine which rotational axes to constrain
        axes = [2] if limit_axes == "z" else [0, 1, 2]

        # Add constraints:
        # ω_axis ≤ ω_max 
        # -ω_axis ≤ ω_max
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
        """
        Construct a quadratic penalty of the form:
        that encourages angular velocity to move the wrist toward a stored
        reference orientation (R0)
        """
        sim = self.env.sim
        qvel_idx = self.env.robots[0].joint_indexes
        ee_site = self.env.robots[0].gripper.important_sites["grip_site"]

        # Initialize initial reference orientation
        if self._R0 is None:
            self._R0 = self._mat_from_site(sim, ee_site)

        # Get the current orientation
        R = self._mat_from_site(sim, ee_site)

        # Compute orientation error
        e_rot = self._so3_log(self._R0.T @ R)

        # Convert error into target angular velocity
        kp = float(self.cfg.get("ori_kp", 2.0))
        omega_ref_full = -kp * e_rot

        # Choose which axes to regulate
        axes = [2] if self.cfg.get("ori_axes", "z") == "z" else [0, 1, 2]

        # Create per-axis weighting matrix W
        W_diag = np.zeros(3)
        for ax in axes:
            W_diag[ax] = float(self.cfg.get("ori_hold_weight", 10.0))

        # Rotational Jacobian at gripper
        Jr_site = np.reshape(sim.data.get_site_jacr(ee_site), (3, -1))
        Jr = Jr_site[:, qvel_idx]

        # Weighted least-squares term:
        # minimize || W (Jr q̇ - ω_ref) ||^2
        # in OSQP:
        # Q = Jrᵀ W² Jr
        # q = - Jrᵀ W² ω_ref
        W2 = np.diag(W_diag**2)
        P_extra = Jr.T @ W2 @ Jr
        q_extra = -(Jr.T @ (W2 @ omega_ref_full))

        return P_extra, q_extra
