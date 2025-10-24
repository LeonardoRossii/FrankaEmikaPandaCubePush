import numpy as np
from .cbf_base import CBFModule

class TableTopCBF(CBFModule):
    def __init__(self, env):
        self.env = env
        self.cz = float(self.env.model.mujoco_arena.table_offset[2])
        self.robot_config = {
            "bodies": {
                "gripper0_eef":        {"alpha": 2.0, "margin": 0.01},
                "gripper0_leftfinger": {"alpha": 2.0, "margin": 0.01},
                "gripper0_rightfinger":{"alpha": 2.0, "margin": 0.01},
                "robot0_link7":        {"alpha": 1.0, "margin": 0.05},
                "robot0_link6":        {"alpha": 1.0, "margin": 0.05},
                "robot0_link5":        {"alpha": 1.0, "margin": 0.05},
                "robot0_link4":        {"alpha": 1.0, "margin": 0.05},
                "robot0_link3":        {"alpha": 1.0, "margin": 0.05},
            },
            "geoms": {
                "robot0_link7_collision": {"alpha": 1.0, "margin": 0.05},
                "robot0_link6_collision": {"alpha": 1.0, "margin": 0.05},
                "robot0_link5_collision": {"alpha": 1.0, "margin": 0.05},
                "robot0_link4_collision": {"alpha": 1.0, "margin": 0.05},
                "robot0_link3_collision": {"alpha": 1.0, "margin": 0.05},
            }
        }

    def _table_plane_cbf(self, env, p, margin):
        z_top = float(env.model.mujoco_arena.table_offset[2])
        h = float(p[2]) - (z_top + margin)
        H = np.array([0.0, 0.0, 1.0], dtype=float)
        return h, H

    def constraints(self, env):
        As, bs = [], []
        sim = env.sim
        model = sim.model
        robot_joint_idx = env.robots[0].joint_indexes

        for body_name, params in self.robot_config["bodies"].items():
            alpha = float(params["alpha"]); margin = float(params["margin"])
            x = sim.data.body_xpos[model.body_name2id(body_name)][:3]
            if abs(x[2] - self.cz) < 10.0:
                h, H = self._table_plane_cbf(env, x, margin)
                jacp = np.reshape(sim.data.get_body_jacp(body_name), (3, -1))
                jacr = np.reshape(sim.data.get_body_jacr(body_name), (3, -1))
                J_full = np.vstack((jacp, jacr))
                J_robot_pos = J_full[:3, robot_joint_idx]
                a = (H @ J_robot_pos).ravel()
                As.append(a.astype(float))
                bs.append(float(-alpha * h))

        for geom_name, params in self.robot_config["geoms"].items():
            alpha = float(params["alpha"]); margin = float(params["margin"])
            x = sim.data.geom_xpos[model.geom_name2id(geom_name)][:3]
            if abs(x[2] - self.cz) < 10.0:
                h, H = self._table_plane_cbf(env, x, margin)
                jacp = np.reshape(sim.data.get_geom_jacp(geom_name), (3, -1))
                jacr = np.reshape(sim.data.get_geom_jacr(geom_name), (3, -1))
                J_full = np.vstack((jacp, jacr))
                J_robot_pos = J_full[:3, robot_joint_idx]
                a = (H @ J_robot_pos).ravel()
                As.append(a.astype(float))
                bs.append(float(-alpha * h))
        return As, bs
