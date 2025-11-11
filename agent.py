import math
import utils
import numpy as np
from metrics import RolloutMetrics
from filter import TableTopCBF, CubeDropCBF, WristRotationCBF, QPFilter

class Agent():
    def __init__(self, env):
        self.env = env
        self.input_size = 17
        self.output_size = self.env.action_dim
        self.A = np.zeros((self.output_size, self.input_size))
        self.b = np.zeros(self.output_size)
        self.cbf_modules = [TableTopCBF(self.env), CubeDropCBF(self.env), WristRotationCBF(self.env)]
        #self.cbf_modules = [CubeDropCBF(self.env)]
        self.safe_filter = QPFilter(env, self.cbf_modules)
        self.action_max = 0.5
        self.action_min = -0.5

    def get_state(self, obs):
        eef_to_cube = obs["eef_pos"]
        eef_to_goal = obs["cube_pos"]      
        cube_to_goal = obs["cube_to_goal"] 
        eef_ori = obs["eef_ori"]
        cube_ori = obs["cube_ori"]
        return np.concatenate([eef_to_cube, cube_to_goal, eef_to_goal, eef_ori, cube_ori])

    def set_weights(self, weights):
        A_size = self.output_size * self.input_size
        self.A[:] = weights[:A_size].reshape((self.output_size, self.input_size))
        self.b[:] = weights[A_size:]  
    
    def get_weights_dim(self):
        return self.input_size * self.output_size + self.output_size
    
    def clip_action(self, action):
        for joint_idx in range(self.env.action_dim-1):
            action[joint_idx] = np.clip(action[joint_idx], self.action_min, self.action_max)
        return action

    def forward(self, x):
        action = np.dot(self.A, x) + self.b
        return self.clip_action(action)

    def evaluate(self, weights, max_n_timesteps, lambdas, gamma=0.99, render=False, save_video = False, video_i=1, plot = False):
        self.set_weights(weights)
        obs = self.env.reset()
        state = self.get_state(obs)
        episode_returns = [0.0] * len(lambdas)
        drop = False
        frames = []
        tracker = RolloutMetrics(log_every=10)
        fixed_action = np.array([0,0.1,0.2,-0.0,0.2,-0.1,-0.2,1])
        ct = False
        rtcasf = []
        cdtasf = []

        for t in range(max_n_timesteps):
            state = self.get_state(obs)
            action = self.forward(state)
            action, efforts = self.safe_filter.apply(action.copy(), self.env)
            #self.env.set_robot_table_collision_avoidance_safety_filter_effort(efforts["TableTopCBF"]["effort"])
            self.env.set_cube_drop_off_table_avoidance_safety_filter_effort(efforts["CubeDropCBF"]["effort"])
            obs, rewards, done, _, = self.env.step(action, lambdas)
            
            if self.env.persist_object_state:
                pos = self.env.sim.data.body_xpos[self.env.cube_body_id].copy()
                quat = self.env.sim.data.body_xquat[self.env.cube_body_id].copy()
                self.env.saved_cube_qpos = np.concatenate([pos, quat])

            if obs["cube_drop"]:
                drop = True

            if(not ct and self.env.check_contact_table()):
                ct = True

            tracker.log_step(t=t, obs=obs)

            if render:
                frame = self.env.sim.render(width=640, height=480, camera_name="frontview")
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
        if save_video:
            utils.save_video(frames, f"video{video_i}.mp4", fps=20)
        self.env.close()

        return episode_returns, metrics, drop, ct