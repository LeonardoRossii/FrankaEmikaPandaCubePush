import math
import numpy as np
from filters import Filter
import torch
import torch.nn as nn
import torch.nn.functional as F

class Agent():
    def __init__(self, env, output_size):
        self.env = env
        self.input_size = 6
        self.output_size = output_size
        self.A = np.zeros((self.output_size, self.input_size))
        self.b = np.zeros(self.output_size)
        self.safe_filter = Filter(self.env)

    def get_state(self, obs):
        eef_to_cube = obs["eef_to_cube"]           
        cube_to_goal = obs["cube_to_goal"] 

        state = np.concatenate([
            eef_to_cube,
            cube_to_goal
        ])
        return state

    def set_weights(self, weights):
        A_size = self.output_size * self.input_size
        self.A[:] = weights[:A_size].reshape((self.output_size, self.input_size))
        self.b[:] = weights[A_size:]  
    
    def get_weights_dim(self):
        return self.input_size * self.output_size + self.output_size
    
    def forward(self, x):
        return np.dot(self.A, x)

    def evaluate(self, weights, params, max_n_timesteps, gamma=0.99):
        obs= self.env.reset()
        state = self.get_state(obs)
        self.set_weights(weights)
        episode_returns = [0.0] * len(params)
        for t in range(max_n_timesteps):
            state = self.get_state(obs)
            action = self.forward(state)
            action = self.safe_filter.apply(action)
            obs, rewards, _, _, = self.env.step(action, params)
            for i in range(len(rewards)):
                episode_returns[i] += rewards[i] * math.pow(gamma, t)
            if self.env.check_success() or self.env.check_failure():
                break
        return episode_returns, obs["cube_drop"]
    
    def episode(self, weight, max_n_timesteps):
        param = [0]
        obs = self.env.reset()
        state = self.get_state(obs)
        self.set_weights(weight)

        eef_to_cube_dist_ = []
        cube_to_goal_dist_ = []
        cube_to_boundary_dist = []

        for t in range(max_n_timesteps):
            state = self.get_state(obs)
            action = self.forward(state)
            action = self.safe_filter.apply(action)
            obs, _, done, _, = self.env.step(action, param)

            if t%10==0:
                eef_to_cube_dist_.append(obs["eef_to_cube_dist"].item())
                cube_to_goal_dist_.append(obs["cube_to_goal_dist"].item())
                cube_to_boundary_dist.append(obs["cube_to_bound_dist"].item())

            if done or self.env.check_success() or self.env.check_failure():
                metrics = {"eef_to_cube_dist": eef_to_cube_dist_,
                        "cube_to_goal_dist": cube_to_goal_dist_,
                        "cube_to_boundary_dist": cube_to_boundary_dist
                        }
                break
        return metrics
    
class NNAgent():
    def __init__(self, env, output_size, hidden_sizes=(32,)):
        self.env = env
        self.input_size = 9
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes
        self.safe_filter = Filter(self.env)

        layers = []
        in_dim = self.input_size
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.Tanh())
            in_dim = h
        layers.append(nn.Linear(in_dim, output_size))
        self.model = nn.Sequential(*layers)

        self.device = torch.device("cpu")
        self.model.to(self.device)

    def get_state(self, obs):
        eef_pos = obs["eef_to_cube"]           
        cube_pos = obs["cube_to_goal"]         
        eef_to_goal = obs["eef_to_goal"] 

        state = np.concatenate([
            eef_pos,
            cube_pos,
            eef_to_goal
        ])
        return state
    
    def get_weights_dim(self):
        return sum(p.numel() for p in self.model.parameters())

    def set_weights(self, flat_weights):
        flat_weights = torch.tensor(flat_weights, dtype=torch.float32, device=self.device)
        idx = 0
        with torch.no_grad():
            for param in self.model.parameters():
                size = param.numel()
                param.copy_(flat_weights[idx:idx + size].view_as(param))
                idx += size

    def get_weights(self):
        return torch.cat([p.detach().flatten() for p in self.model.parameters()]).cpu().numpy()

    def forward(self, x):
        x = torch.tensor(x, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            action = self.model(x)
        return action.cpu().numpy()

    def evaluate(self, weights, params, max_n_timesteps, gamma=0.99):
        obs = self.env.reset()
        state = self.get_state(obs)
        self.set_weights(weights)

        episode_returns = [0.0] * len(params)
        for t in range(max_n_timesteps):
            state = self.get_state(obs)
            action = self.forward(state)
            action = self.safe_filter.apply(action)

            obs, rewards, _, _, = self.env.step(action, params)
            for i in range(len(rewards)):
                episode_returns[i] += rewards[i] * (gamma ** t)

            if self.env.check_success() or self.env.check_failure():
                break

        return episode_returns, obs["cube_drop"]

    def episode(self, weight, max_n_timesteps):
        param = [0]
        obs = self.env.reset()
        state = self.get_state(obs)
        self.set_weights(weight)

        eef_to_cube_dist_ = []
        cube_to_goal_dist_ = []
        cube_to_boundary_dist = []

        for t in range(max_n_timesteps):
            state = self.get_state(obs)
            action = self.forward(state)
            action = self.safe_filter.apply(action)
            obs, _, done, _, = self.env.step(action, param)

            if t % 10 == 0:
                eef_to_cube_dist_.append(obs["eef_to_cube_dist"].item())
                cube_to_goal_dist_.append(obs["cube_to_goal_dist"].item())
                cube_to_boundary_dist.append(obs["cube_to_bound_dist"].item())

            if done or self.env.check_success() or self.env.check_failure():
                metrics = {
                    "eef_to_cube_dist": eef_to_cube_dist_,
                    "cube_to_goal_dist": cube_to_goal_dist_,
                    "cube_to_boundary_dist": cube_to_boundary_dist
                }
                break
        return metrics
