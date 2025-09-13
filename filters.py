import osqp
import numpy as np
import scipy.sparse as sp

class Filter:
    def __init__(self, env):
        self.env = env
        self.min = 0.05

    def apply(self, action):
        dist = self.env.get_cube_bound_dist()
        dim = self.env.action_dim
        if dist<self.min:
            action = np.zeros(dim)
        return action
    
class FilterCBF:
    def __init__(self, env, n_constraints=4, ctrl_range= 1.0):
        self.env = env
        self.n = self.env.action_dim 
        self.m = n_constraints
        self.ctrl_range = ctrl_range
    
    def apply(self, action):
        pass