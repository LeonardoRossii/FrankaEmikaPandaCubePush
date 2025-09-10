import math
import numpy as np

class Agent():
    def __init__(self, env, input_size, output_size):
        self.env = env
        self.input_size = input_size
        self.output_size = output_size
        self.A = np.zeros((output_size, input_size))
        self.b = np.zeros(output_size)

    def get_state(self,  obs):
        return np.array([obs["eef_to_cube_pos"], obs["cube_to_goal_pos"]])

    def set_weights(self, weights):
        A_size = self.output_size * self.input_size
        self.A[:] = weights[:A_size].reshape((self.output_size, self.input_size))  
        self.b[:] = weights[A_size:]  
    
    def get_weights_dim(self):
        return self.input_size * self.output_size + self.output_size
    
    def forward(self, x):
        return np.dot(self.A, x) + self.b

    def evaluate(self, weights, max_n_timesteps, gamma=0.99):
        obs= self.env.reset()
        state = self.get_state(obs)
        self.set_weights(weights)
        episode_return = 0.0
        for t in range(max_n_timesteps):
            state = self.get_state(obs)
            action = self.forward(state)
            obs, reward, _, _, = self.env.step(action)
            episode_return += reward * math.pow(gamma, t)
            if self.env.check_success() or self.env.check_failure():
                break
        return episode_return