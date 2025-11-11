import spec
import numpy as np
from robosuite.models.arenas import TableArena   
from robosuite.models.objects import BoxObject   
from robosuite.models.tasks import ManipulationTask   
from robosuite.utils.mjcf_utils import CustomMaterial   
from robosuite.utils.observables import Observable, sensor   
from robosuite.utils.placement_samplers import UniformRandomSampler
from robosuite.environments.manipulation.single_arm_env import SingleArmEnv 


class Push(SingleArmEnv):
    def __init__(self, 
                 robots,
                 ...,
                 hard_reset=True,
                 ...,
                 seed=42,
                 persist_object_state=True,   # <--- NEW
                 ):
        ...
        self.persist_object_state = persist_object_state
        self._saved_cube_qpos = None      # 7D [x,y,z, qw,qx,qy,qz]
        ...

    ...
    def _reset_internal(self):
        super()._reset_internal()

        # If we want stateful resets AND we have a saved cube pose, restore it.
        if self.persist_object_state and (self._saved_cube_qpos is not None):
            # Restore cube free-joint qpos (pos+quat), and zero its velocity
            self.sim.data.set_joint_qpos(self.cube.joints[0], self._saved_cube_qpos.copy())
            try:
                # Zero joint velocity if available on your mujoco binding
                self.sim.data.set_joint_qvel(self.cube.joints[0], np.zeros(6))
            except Exception:
                pass  # older bindings may not expose set_joint_qvel

        else:
            # First episode (or persistence disabled): use your current placement logic
            if not self.deterministic_reset:
                placements = self.placement_initializer.sample()
                for obj_pos, obj_quat, obj in placements.values():
                    self.sim.data.set_joint_qpos(
                        obj.joints[0], np.concatenate([np.array(obj_pos), np.array(obj_quat)])
                    )

        # (re)compute goal from the (possibly restored) cube pose
        cube_xy = self.sim.data.body_xpos[self.cube_body_id][:2].copy()
        self.goal_xy = cube_xy + self.goal_pos_offset
        self.table_robot_collision_avoidance_safety_filter_effort = 0.0
        self.cube_drop_off_table_avoidance_safety_filter_effort = 0.0
        self.sim.forward()

    # --- NEW: save cube pose continuously so the next reset can pick it up
    def step(self, action):
        obs, reward, done, info = super().step(action)

        if self.persist_object_state:
            # Save current cube pos+quat as a 7D free-joint qpos vector
            pos = self.sim.data.body_xpos[self.cube_body_id].copy()
            quat = self.sim.data.body_xquat[self.cube_body_id].copy()   # [w, x, y, z] in MuJoCo
            self._saved_cube_qpos = np.concatenate([pos, quat])

        return obs, reward, done, info
