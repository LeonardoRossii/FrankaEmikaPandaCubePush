import numpy as np
import cvxpy as cp

class OSPMPCFilter:
    def __init__(self, env,
                 table_xy, z_limits,
                 u_pos_max=(1, 1, 1),
                 u_ori_max=(0.05, 0.05, 0.05),
                 N=10):
        """
        table_xy: (xmin, xmax, ymin, ymax)
        z_limits: (zmin, zmax)
        u_*_max: per-axis increment caps (meters, radians)
        """
        self.env = env
        self.N = N
        self.xmin, self.xmax, self.ymin, self.ymax = table_xy
        self.zmin, self.zmax = z_limits
        self.u_pos_max = np.array(u_pos_max, float)
        self.u_ori_max = np.array(u_ori_max, float)

        self.nx = 6  # [px,py,pz, thx,thy,thz]
        self.nu = 6  # increment in same order

        # Variables (built once, reused each tick)
        self.X = cp.Variable((self.nx, N+1))
        self.U = cp.Variable((self.nu, N))

        # Params set each solve
        self.x0_par    = cp.Parameter(self.nx)
        self.U_nom_par = cp.Parameter((self.nu, N))
        self.X_ref_par = cp.Parameter((self.nx, N+1))

        # Costs (tune to taste)
        Qp = np.diag([0.1, 0.1, 0.2])           # keep position near ref
        Qo = np.diag([1, 1, 1])             # keep orientation near ref
        Q  = np.block([[Qp, np.zeros((3,3))],
                       [np.zeros((3,3)), Qo]])
        R  = np.diag([1,1,1, 0.2,0.2,0.2])  # stay close to nominal increments

        constr = [self.X[:,0] == self.x0_par]
        cost = 0
        for k in range(N):
            # Integrator dynamics: x_{k+1} = x_k + u_k
            constr += [ self.X[:,k+1] == self.X[:,k] + self.U[:,k] ]

            # Workspace on position
            constr += [
                self.xmin <= self.X[0,k], self.X[0,k] <= self.xmax,
                self.ymin <= self.X[1,k], self.X[1,k] <= self.ymax,
                self.zmin <= self.X[2,k], self.X[2,k] <= self.zmax,
            ]

            # Increment caps
            up = self.U[0:3, k]; uo = self.U[3:6, k]
            constr += [
                -self.u_pos_max <= up, up <= self.u_pos_max,
                -self.u_ori_max <= uo, uo <= self.u_ori_max,
            ]

            dx = self.X[:,k] - self.X_ref_par[:,k]
            du = self.U[:,k] - self.U_nom_par[:,k]
            cost += cp.quad_form(dx, Q) + cp.quad_form(du, R)

        dxN = self.X[:,N] - self.X_ref_par[:,N]
        cost += cp.quad_form(dxN, Q)

        self.prob = cp.Problem(cp.Minimize(cost), constr)

    # --- helpers
    def _eef_pos(self):
        # Use the ROBOT EEF position (world frame)
        return np.array(self.env.sim.data.site_xpos[self.env.robots[0].eef_site_id], float)[:3]

    def _eef_theta_err(self):
        # Keep current orientation → zero error vector.
        # (If you have a desired quaternion q_d, compute small axis-angle error here.)
        return np.zeros(3, float)

    def _x0(self):
        p = self._eef_pos()
        th = self._eef_theta_err()
        return np.concatenate([p, th])

    def apply(self, action, debug: bool = False):
        """Filter the first 6 dims of 'action' (pos+ori increments) with MPC + diagnostics.
        IMPORTANT: X_ref is built by rolling out the nominal increments, so the MPC follows u_nom unless constraints bind.
        """
        act = np.array(action, dtype=float, copy=True)

        # 1) Nominal increments
        u_nom = np.zeros(6, dtype=float)
        n_copy = min(6, act.shape[0])
        u_nom[:n_copy] = act[:n_copy]

        # 2) Horizon warm-start by repetition
        U_nom = np.tile(u_nom.reshape(-1, 1), (1, self.N))  # 6 x N

        # 3) Build state reference by rolling out U_nom: x_{k+1} = x_k + u_nom_k
        x0 = self._x0()
        X_ref = np.tile(x0.reshape(-1, 1), (1, self.N + 1))
        for k in range(self.N):
            X_ref[:, k + 1] = X_ref[:, k] + U_nom[:, k]

        # 4) Solve QP
        self.x0_par.value    = x0
        self.U_nom_par.value = U_nom
        self.X_ref_par.value = X_ref

        try:
            self.prob.solve(
                solver=cp.OSQP,
                warm_start=True,
                eps_abs=1e-6,
                eps_rel=1e-6,
                max_iter=20000,
            )
            status = getattr(self.prob, "status", "unknown")
            stats = getattr(self.prob, "solver_stats", None)
            solve_time = getattr(stats, "solve_time", None) if stats else None
            obj = getattr(self.prob, "value", None)
        except Exception as e:
            status, solve_time, obj = f"exception: {type(e).__name__}", None, None

        if getattr(self, "U", None) is None or self.U.value is None or not np.all(np.isfinite(self.U.value)):
            u_safe = np.zeros(6, dtype=float)
        else:
            u_safe = np.array(self.U.value[:, 0]).ravel().astype(float)

        # 5) Diagnostics
        try:
            eef = np.array(self.env.sim.data.site_xpos[self.env.robots[0].eef_site_id], float)
            eef_xyz = eef[:3]
        except Exception:
            eef_xyz = np.array([np.nan, np.nan, np.nan])

        xmin, xmax, ymin, ymax = self.xmin, self.xmax, self.ymin, self.ymax
        dx_min = eef_xyz[0] - xmin
        dx_max = xmax - eef_xyz[0]
        dy_min = eef_xyz[1] - ymin
        dy_max = ymax - eef_xyz[1]

        def sat_ratio(val, lim):
            lim = np.asarray(lim, float); val = np.asarray(val, float)
            with np.errstate(divide='ignore', invalid='ignore'):
                r = np.abs(val) / np.maximum(lim, 1e-12)
            return float(np.nanmax(r))

        sat_pos = sat_ratio(u_safe[:3], self.u_pos_max)
        sat_ori = sat_ratio(u_safe[3:], self.u_ori_max)

        self._last_diag = {
            "status": status,
            "objective": obj,
            "solve_time_s": solve_time,
            "eef_xyz": eef_xyz,
            "u_nom": u_nom.copy(),
            "u_safe": u_safe.copy(),
            "sat_pos_max_ratio": sat_pos,
            "sat_ori_max_ratio": sat_ori,
            "dist_to_walls": {"x_min": dx_min, "x_max": dx_max, "y_min": dy_min, "y_max": dy_max},
        }

        if debug:
            print(
                "[mpc_filter]"
                f" status={status}"
                f" t={solve_time:.6f}s" if isinstance(solve_time, (int, float)) else " t=?",
                f" eef=({eef_xyz[0]:+.3f},{eef_xyz[1]:+.3f},{eef_xyz[2]:+.3f})" if np.all(np.isfinite(eef_xyz)) else " eef=(nan,nan,nan)",
                f" walls(dx=[{dx_min:+.3f},{dx_max:+.3f}], dy=[{dy_min:+.3f},{dy_max:+.3f}])",
                f" u_nom=({u_nom[0]:+.3f},{u_nom[1]:+.3f},{u_nom[2]:+.3f}|{u_nom[3]:+.3f},{u_nom[4]:+.3f},{u_nom[5]:+.3f})",
                f" u_safe=({u_safe[0]:+.3f},{u_safe[1]:+.3f},{u_safe[2]:+.3f}|{u_safe[3]:+.3f},{u_safe[4]:+.3f},{u_safe[5]:+.3f})",
                f" sat(pos={sat_pos:.2f}, ori={sat_ori:.2f})",
            )

        # 6) Splice back
        act[:6] = u_safe
        return act
