import numpy as np
from .cbf_base import CBFModule

class TableTopCBF(CBFModule):
    """
    Control Barrier Function (CBF) enforcing that the robot does not move below the tabletop surface.
    This CBF checks the height (z-position) of various robot points.
    It constructs inequality constraints that prevent the robot from violating a minimum clearance margin above the tabletop.
    """
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
        
        #Compute the CBF condition for 
        #  minimum height above the table
        # Get the toptable z coordinate
        z_top = float(env.model.mujoco_arena.table_offset[2])

        # Compute CBF function with margin
        h = float(p[2]) - (z_top + margin)

        # Compute the gradient with respect to eef position
        H = np.array([0.0, 0.0, 1.0], dtype=float)
        return h, H

    def constraints(self, env):
        As, bs = [], []
        sim = env.sim
        model = sim.model

        # Indices of actuated robot joints 
        # Used to extract proper Jacobian columns
        robot_joint_idx = env.robots[0].joint_indexes

        # For each body
        for body_name, params in self.robot_config["bodies"].items():
            
            # Get the associated parameters
            alpha = float(params["alpha"])
            margin = float(params["margin"])

            # Get the associated world-frame position 
            x = sim.data.body_xpos[model.body_name2id(body_name)][:3]

            # Compute the barrier value and gradient 
            h, H = self._table_plane_cbf(env, x, margin)

            # Retrieve Jacobians
            # Reshape from 1D vector to matrix 3xn (n= number of joints)
            jacp = np.reshape(sim.data.get_body_jacp(body_name), (3, -1))
            jacr = np.reshape(sim.data.get_body_jacr(body_name), (3, -1))

            # Stack full Jacobian
            # Shape matrix 6xn
            J_full = np.vstack((jacp, jacr))

            # Select the first three rows 3xn translational Jacobian
            J_robot_pos = J_full[:3, robot_joint_idx]

            # Compute A
            # Flatten to 1D row vector (QP expect this form)
            # H^T J_p
            a = (H @ J_robot_pos).ravel()

            # Store the constraint row in list of all constraints
            As.append(a.astype(float))

            # ompute the matching lower-bound b=−αh
            bs.append(float(-alpha * h))

        # For each geom to exactly the same done for each body above
        for geom_name, params in self.robot_config["geoms"].items():
            alpha = float(params["alpha"])
            margin = float(params["margin"])
            x = sim.data.geom_xpos[model.geom_name2id(geom_name)][:3]
            h, H = self._table_plane_cbf(env, x, margin)
            jacp = np.reshape(sim.data.get_geom_jacp(geom_name), (3, -1))
            jacr = np.reshape(sim.data.get_geom_jacr(geom_name), (3, -1))
            J_full = np.vstack((jacp, jacr))
            J_robot_pos = J_full[:3, robot_joint_idx]
            a = (H @ J_robot_pos).ravel()
            As.append(a.astype(float))
            bs.append(float(-alpha * h))

        return As, bs
