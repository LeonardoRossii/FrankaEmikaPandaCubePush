import osqp
import numpy as np
import scipy.sparse as sp

class Filter:
    def __init__(self, env):
        self.env = env
        self.min = 0.03

    def apply(self, action):
        dist = self.env.get_cube_bound_dist()
        dim = self.env.action_dim
        if dist<self.min:
            action = np.zeros(dim)
        return action
    

class FilterCBF:
    def __init__(self, env, alpha=0.5):
        self.env = env
        self.alpha = float(alpha)

        self.dt = env.control_freq

        half_x = float(self.env.table_full_size[0]) / 2.0
        half_y = float(self.env.table_full_size[1]) / 2.0
        self.xmin, self.xmax = -half_x, +half_x
        self.ymin, self.ymax = -half_y, +half_y

        self.n = 2

        P = sp.eye(self.n, format="csc")
        q0 = np.zeros(self.n)

        A = sp.eye(self.n, format="csc")
        l0 = -np.ones(self.n) * np.inf
        u0 =  np.ones(self.n) * np.inf

        self._solver = osqp.OSQP()
        self._solver.setup(P=P, q=q0, A=A, l=l0, u=u0,
                           verbose=False, warm_start=True, polish=False,
                           eps_abs=1e-6, eps_rel=1e-6, max_iter=20000)

        self.pos = None
        self._last_x = np.zeros(self.n)

    def apply(self, action):
        
        nom = action[:2]         
        
        pos = self.env.get_cube_pos()[:2]

        bmin = np.array([self.xmin, self.ymin])
        bmax = np.array([self.xmax, self.ymax])

        lo = -self.alpha * self.dt * (pos - bmin)
        hi =  self.alpha * self.dt * (bmax - pos)

        q = -nom
        self._solver.update(q=q, l=lo, u=hi)
        self._solver.warm_start(x=self._last_x)
        res = self._solver.solve()

        if res.info.status_val == osqp.constant('OSQP_SOLVED') and res.x is not None:
            d_safe = res.x
        else:
            d_safe = np.zeros(self.n + 1) 
        
        self._last_x = d_safe

        action[:2] = d_safe
        return action
