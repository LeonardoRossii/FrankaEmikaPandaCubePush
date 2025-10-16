import os
import math
import utils
import numpy as np
from mpc import OSPMPCFilter
from filters import FilterCBF
from metrics import RolloutMetrics
from table_collision import TableCollisionFilter

class Agent():
    def __init__(self, env):
        self.env = env
        self.input_size = 9
        self.output_size = self.env.action_dim
        self.A = np.zeros((self.output_size, self.input_size))
        self.b = np.zeros(self.output_size)
        self.safe_filter = TableCollisionFilter(self.env)

    def get_state(self, obs):
        eef_to_cube = obs["eef_to_cube"]
        eef_to_goal = obs["eef_to_goal"]      
        cube_to_goal = obs["cube_to_goal"] 
        return np.concatenate([eef_to_cube, cube_to_goal, eef_to_goal])

    def set_weights(self, weights):
        A_size = self.output_size * self.input_size
        self.A[:] = weights[:A_size].reshape((self.output_size, self.input_size))
        self.b[:] = weights[A_size:]  
    
    def get_weights_dim(self):
        return self.input_size * self.output_size + self.output_size
    
    def forward(self, x):
        return np.dot(self.A, x) + self.b

    def evaluate(self, weights, max_n_timesteps, lambdas, gamma=0.99, render=False, video_i=1):
        self.set_weights(weights)
        obs = self.env.reset()
        state = self.get_state(obs)
        episode_returns = [0.0] * len(lambdas)
        drop = False
        frames = []
        tracker = RolloutMetrics(log_every=10)
        fixed_action = np.array([0.0,0.5,0,0,0,0,0,0])

        for t in range(max_n_timesteps):
            state = self.get_state(obs)
            action = self.forward(state)
            action = self.safe_filter.apply(fixed_action.copy())
            obs, rewards, done, _, = self.env.step(action, lambdas)
            if obs["cube_drop"]:
                drop = True

            tracker.log_step(t=t, obs=obs)

            if render:
                frame = self.env.sim.render(width=640, height=480, camera_name="sideview")
                frame = frame[::-1, :, :]
                frames.append(frame)

            for i in range(len(rewards)):
                episode_returns[i] += float(rewards[i]) * math.pow(gamma, t)

            if done or self.env.check_success() or self.env.check_failure():
                accomplished = bool(self.env.check_success())
                metrics = tracker.to_dict(accomplished)
                break
            if render:
                self.env.render()
        if render:
            utils.save_video(frames, f"video{video_i}.mp4", fps=20)

        self.env.close()
        return episode_returns, metrics, drop