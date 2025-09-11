import math
import numpy as np

class Agent():
    def __init__(self, env, output_size):
        self.env = env
        self.input_size = 2
        self.output_size = output_size
        self.A = np.zeros((self.output_size, self.input_size))
        self.b = np.zeros(self.output_size)

    def get_state(self,  obs):
        return np.array([obs["eef_to_cube_dist"], obs["cube_to_goal_dist"]])

    def set_weights(self, weights):
        A_size = self.output_size * self.input_size
        self.A[:] = weights[:A_size].reshape((self.output_size, self.input_size))  
        self.b[:] = weights[A_size:]  
    
    def get_weights_dim(self):
        return self.input_size * self.output_size + self.output_size
    
    def forward(self, x):
        return np.dot(self.A, x) + self.b

    def evaluate(self, weights, params, max_n_timesteps, gamma=0.99):
        obs= self.env.reset()
        state = self.get_state(obs)
        self.set_weights(weights)
        episode_returns = [0.0] * len(params)
        for t in range(max_n_timesteps):
            state = self.get_state(obs)
            action = self.forward(state)
            obs, rewards, _, _, = self.env.step(action, params)
            for i in range(len(rewards)):
                episode_returns[i] += rewards[i] * math.pow(gamma, t)
            if self.env.check_success() or self.env.check_failure():
                break
        return episode_returns
    
    def episode(self, weight, max_n_timesteps):
        param = [0]
        obs = self.env.reset()
        state = self.get_state(obs)
        self.set_weights(weight)

        eef_to_cube_dist_ = []
        cube_to_goal_dist_ = []

        for t in range(max_n_timesteps):
            state = self.get_state(obs)
            action = self.forward(state)
            obs, _, done, _, = self.env.step(action, param)

            eef_to_cube_dist_.append(obs["eef_to_cube_dist"].item())
            cube_to_goal_dist_.append(obs["cube_to_goal_dist"].item())

            if done or self.env.check_success() or self.env.check_failure():
                dict = {"eef_to_cube_dist": eef_to_cube_dist_,"cube_to_goal_dist": cube_to_goal_dist_}
                break
        return dict