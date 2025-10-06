import osqp
import numpy as np
import scipy.sparse as sp

class Filter:
    def __init__(self, env):
        self.env = env
        self.min = 0.03
        self.effort = 0.0

    def apply(self, action):
        dist = self.env.get_cube_bound_dist()
        dim = self.env.action_dim
        if dist<self.min:
            action = np.zeros(dim)
        return action
    
class FilterCBF:
    def __init__(self, env, alpha=1.0):
        self.env = env
        self.alpha = float(alpha)

        self.dt = 1.0/env.control_freq

        half_x = float(self.env.table_full_size[0]) / 2.0
        half_y = float(self.env.table_full_size[1]) / 2.0

        safe_margin = 0.1*half_x
        
        self.xmin, self.xmax = -half_x+safe_margin, +half_x-safe_margin
        self.ymin, self.ymax = -half_y+safe_margin, +half_y-safe_margin

        self.xspan = self.xmax - self.xmin
        self.yspan = self.ymax - self.ymin

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

    def step(self, dist, span):
        s = np.clip(dist / max(span, 1e-9), 0.0, 1.0)
        step = np.tanh(self.alpha * s)              
        return min(step, dist)
    
    def apply(self, action):
        
        act = action.copy()
        nom = act[:2]
        
        pos = self.env.get_cube_pos()[:2]
        
        d_to_min = np.array([pos[0] - self.xmin, pos[1] - self.ymin])
        d_to_max = np.array([self.xmax - pos[0], self.ymax - pos[1]])
        
        spans = np.array([(self.xmax - self.xmin)/2, (self.ymax - self.ymin)/2])

        lo = np.zeros(2)
        hi = np.zeros(2)

        for i in range(2):
            hi[i] =  self.step(d_to_max[i], spans[i])
            lo[i] = -self.step(d_to_min[i], spans[i])
    
        q = -nom

        self._solver.update(q=q, l=lo, u=hi)
        self._solver.warm_start(x=self._last_x)
        res = self._solver.solve()

        if res.info.status_val == osqp.constant('OSQP_SOLVED') and res.x is not None:
            d_safe = res.x
        else:
            d_safe = np.zeros(self.n + 1) 

        self._last_x = d_safe

        self.env.safety_filter_effort = np.linalg.norm(action[:2] - d_safe)

        act[:2] = d_safe
        return act