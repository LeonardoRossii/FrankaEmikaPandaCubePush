import os
import math
import utils
import numpy as np
from filters import FilterCBF

class Agent():
    def __init__(self, env):
        self.env = env
        self.input_size = 9
        self.output_size = self.env.action_dim
        self.A = np.zeros((self.output_size, self.input_size))
        self.b = np.zeros(self.output_size)
        self.safe_filter = FilterCBF(self.env)

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

    def evaluate(self, weights, max_n_timesteps, gamma=0.99, render=False, trajectory=False):
        # Set controller parameters
        self.set_weights(weights)
        # Reset environment and safety filter
        obs = self.env.reset()
        state = self.get_state(obs)
        
        episode_return = 0.0
        frames = []
        metrics = None

        # Helper for quaternion tilt (degrees) relative to vertical
        def cube_tilt_deg_from_quat(q):
            # MuJoCo xquat convention: [w, x, y, z]
            w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
            # R[2,2] = 1 - 2(x^2 + y^2)
            r22 = 1.0 - 2.0 * (x * x + y * y)
            r22 = max(-1.0, min(1.0, r22))  # numerical safety
            return float(np.degrees(np.arccos(r22)))

        # Storage for rollout metric evolution
        rollout_cube_to_goal_dist = []
        rollout_eef_to_cube_dist = []
        rollout_eef_height_rel = []
        rollout_eef_behind_alignment = []
        rollout_cube_tilt_deg = []
        rollout_cube_to_bound_dist = []
        rollout_cube_drop = []
        rollout_cube_speed_xy = []
        rollout_push_progress = []
        rollout_contact_est = []

        # Running references for deltas
        prev_cube_pos = None
        prev_cube_to_goal = None

        # Sampling stride for logging (reduce size but keep evolution)
        log_stride = 5

        for t in range(max_n_timesteps):
            state = self.get_state(obs)
            action = self.forward(state)
            # Apply safety filter (reactive shielding if needed)
            action = self.safe_filter.apply(action)

            obs, rewards, done, _, = self.env.step(action, [0])

            if render:
                frame = self.env.sim.render(width=640, height=480, camera_name="sideview")
                frame = frame[::-1, :, :]
                frames.append(frame)

            # Compute and log metrics at chosen stride
            if (not trajectory) or (trajectory and (t % log_stride == 0)):
                # Distances
                cube_to_goal_vec = np.asarray(obs["cube_to_goal"])
                eef_to_cube_vec = np.asarray(obs["eef_to_cube"])
                cube_pos = np.asarray(obs["cube_pos"])
                cube_quat = np.asarray(obs["cube_quat"])

                cube_to_goal_norm = float(np.linalg.norm(cube_to_goal_vec))
                eef_to_cube_norm = float(np.linalg.norm(eef_to_cube_vec))
                eef_height_rel = float(eef_to_cube_vec[2])  # eef height relative to cube center (positive = above)

                # Alignment: eef behind cube relative to goal (1 good, 0 bad)
                g_xy = cube_to_goal_vec[:2]
                e_xy = eef_to_cube_vec[:2]
                g_norm = np.linalg.norm(g_xy)
                e_norm = np.linalg.norm(e_xy)
                if g_norm > 1e-8 and e_norm > 1e-8:
                    cos_angle = float(np.dot(e_xy, g_xy) / (e_norm * g_norm))
                    # negative cos means eef behind the cube relative to goal direction
                    behind_alignment = max(0.0, -cos_angle)  # in [0,1]
                else:
                    behind_alignment = 0.0

                # Contact estimate: close lateral approach with small vertical offset
                contact = 1 if (e_norm < 0.04 and abs(eef_height_rel) < 0.05) else 0

                # Cube motion (xy speed per log step)
                if prev_cube_pos is not None:
                    cube_speed_xy = float(np.linalg.norm((cube_pos - prev_cube_pos)[:2]))
                else:
                    cube_speed_xy = 0.0

                # Goal progress (positive if getting closer)
                if prev_cube_to_goal is not None:
                    push_progress = float(prev_cube_to_goal - cube_to_goal_norm)
                else:
                    push_progress = 0.0

                # Safety-related metrics
                cube_to_bound = float(obs["cube_to_bound_dist"])
                # cube_drop may be bool; map to 0/1
                cube_drop_flag = int(bool(obs["cube_drop"])) if "cube_drop" in obs else 0
                tilt_deg = cube_tilt_deg_from_quat(cube_quat)

                # Append to logs
                rollout_cube_to_goal_dist.append(cube_to_goal_norm)
                rollout_eef_to_cube_dist.append(eef_to_cube_norm)
                rollout_eef_height_rel.append(eef_height_rel)
                rollout_eef_behind_alignment.append(behind_alignment)
                rollout_cube_tilt_deg.append(tilt_deg)
                rollout_cube_to_bound_dist.append(cube_to_bound)
                rollout_cube_drop.append(cube_drop_flag)
                rollout_cube_speed_xy.append(cube_speed_xy)
                rollout_push_progress.append(push_progress)
                rollout_contact_est.append(int(contact))

                # Update prev references
                prev_cube_pos = cube_pos.copy()
                prev_cube_to_goal = cube_to_goal_norm

            episode_return += float(rewards) * math.pow(gamma, t)

            # Early termination and metrics packaging
            if done or self.env.check_success() or self.env.check_failure():
                metrics = {
                    # Core task metrics (evolution)
                    "rollout_cube_to_goal_dist": rollout_cube_to_goal_dist,
                    "rollout_eef_to_cube_dist": rollout_eef_to_cube_dist,
                    "rollout_contact_est": rollout_contact_est,            # 0/1 estimate of finger-object contact
                    "rollout_push_progress": rollout_push_progress,        # positive values indicate progress
                    "rollout_cube_speed_xy": rollout_cube_speed_xy,        # cube motion in plane
                    # Safety-related metrics (evolution)
                    "rollout_cube_to_bound_dist": rollout_cube_to_bound_dist,
                    "rollout_cube_drop": rollout_cube_drop,                # 0/1 potential irreversible event (falling)
                    "rollout_cube_tilt_deg": rollout_cube_tilt_deg,        # toppling/pressing risk
                    "rollout_eef_height_rel": rollout_eef_height_rel,      # eef vertical relative to cube center
                    "rollout_eef_behind_alignment": rollout_eef_behind_alignment,  # 0..1 (1 = well-behind to push)
                    # Episode outcomes / summaries
                    "rollout_accomplished": bool(self.env.check_success()),
                    "rollout_failure": bool(self.env.check_failure()),
                }
                break

            if render:
                self.env.render()

        if render:
            utils.save_video(frames, "push_demo.mp4", fps=20)

        self.env.close()
        return episode_return, metrics