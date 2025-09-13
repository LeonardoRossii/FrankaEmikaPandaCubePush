import numpy as np
from robosuite.models.arenas import TableArena   
from robosuite.models.objects import BoxObject   
from robosuite.models.tasks import ManipulationTask   
from robosuite.utils.mjcf_utils import CustomMaterial   
from robosuite.utils.observables import Observable, sensor   
from robosuite.utils.placement_samplers import UniformRandomSampler
from robosuite.environments.manipulation.single_arm_env import SingleArmEnv 
import spec

class Push(SingleArmEnv):
    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        initialization_noise=None,
        table_full_size=(0.8, 0.15, 0.05),
        table_friction=(0.5, 5e-3, 1e-4),
        use_object_obs=True,
        use_camera_obs = False,
        reward_scale=None,
        reward_shaping=True,
        placement_initializer=None,
        has_renderer=True,
        has_offscreen_renderer=True,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=25,
        horizon=250,
        ignore_done=False,
        hard_reset=True,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        camera_segmentations=None, 
        renderer="mujoco",
        renderer_config=None,
    ):
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping
        self.use_object_obs = use_object_obs
        self.placement_initializer = placement_initializer

        self.table_offset = np.array((0, 0, 0.95))
        self.goal_pos_offset = np.array([0.2, 0])

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            mount_types="default",
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            camera_segmentations=camera_segmentations,
            renderer=renderer,
            renderer_config=renderer_config,
        )

    def reward(self, action=None, param = None):
        return spec.get_reward(self, action, param)
    
    def check_success(self):
        return spec.get_success_condition(self)
    
    def check_failure(self):
        return spec.get_failure_condition(self)
    
    def get_eef_pos(self):
        return self.sim.data.site_xpos[self.robots[0].eef_site_id]
    
    def get_cube_pos(self):
        return self.sim.data.body_xpos[self.cube_body_id] 
    
    def get_goal_pos(self):
        return np.array([self.goal_xy[0], self.goal_xy[1], self.model.mujoco_arena.table_offset[2] + 0.001], dtype=float)
    
    def get_cube_bound_dist(self):
        return min(
            np.linalg.norm(self.sim.data.body_xpos[self.cube_body_id][1]-self.table_full_size[1]/2),
            np.linalg.norm(self.sim.data.body_xpos[self.cube_body_id][1]+self.table_full_size[1]/2)
        )
     
    def check_contact_table(self):
        table_contact= False
        for contact in self.sim.data.contact:
            geom1 = self.sim.model.geom_id2name(contact.geom1)
            geom2 = self.sim.model.geom_id2name(contact.geom2)
            if ("table" in geom1 or "table" in geom2) and ("gripper" in geom1 or "gripper" in geom2 or "robot" in geom1 or "robot" in geom2):
                table_contact= True
        return table_contact
    
    def check_contact_cube(self):
        cube_contact = False
        for contact in self.sim.data.contact:
            geom1 = self.sim.model.geom_id2name(contact.geom1)
            geom2 = self.sim.model.geom_id2name(contact.geom2)
            if ("cube_g0" in geom1 or "cube_g0" in geom2) and ("gripper0_finger1_collision" in geom1 or "gripper0_finger1_collision" in geom2 or "gripper0_finger2_collision" in geom1 or "gripper0_finger2_collision" in geom2):
                cube_contact= True
        return cube_contact


    def _load_model(self):
        super()._load_model()
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )
        mujoco_arena.set_origin([0, 0, 0])
        tex_attrib = {"type": "cube"}
        mat_attrib = {"texrepeat": "1 1", "specular": "0.4", "shininess": "0.1"}
        redwood = CustomMaterial(
            texture="WoodRed",
            tex_name="redwood",
            mat_name="redwood_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        self.cube = BoxObject(
            name="cube",
            size_min=[0.020, 0.020, 0.020],
            size_max=[0.020, 0.020, 0.020],
            rgba=[1, 0, 0, 1],
            material=redwood,
        )
        if self.placement_initializer is not None:
            self.placement_initializer.reset()
            self.placement_initializer.add_objects(self.cube)
        else:
            self.placement_initializer = UniformRandomSampler(
                name="ObjectSampler",
                mujoco_objects=self.cube,
                x_range=[0.0, 0.0],
                y_range=[0.0, 0.0],
                rotation=0,
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.01,
            )
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=self.cube,
        )

    def _setup_references(self):
        super()._setup_references()
        self.cube_body_id = self.sim.model.body_name2id(self.cube.root_body)

    def _setup_observables(self):
        observables = super()._setup_observables()

        if self.use_object_obs:
            pf = self.robots[0].robot_model.naming_prefix
            modality = "object"

            @sensor(modality=modality)
            def cube_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.cube_body_id])

            @sensor(modality=modality)
            def goal_pos(obs_cache):
                table_z = self.model.mujoco_arena.table_offset[2]
                goal_pos =  np.array([self.goal_xy[0], self.goal_xy[1], table_z + 0.001], dtype=float)
                return goal_pos

            @sensor(modality=modality)
            def eef_to_cube_dist(obs_cache):
                if f"{pf}eef_pos" in obs_cache and "cube_pos" in obs_cache:
                    return np.linalg.norm(obs_cache[f"{pf}eef_pos"] - obs_cache["cube_pos"])
                return 0

            @sensor(modality=modality)
            def cube_to_goal_dist(obs_cache):
                if "cube_pos" in obs_cache and "goal_pos" in obs_cache:
                    return np.linalg.norm(obs_cache["goal_pos"] - obs_cache["cube_pos"])
                return 0

            @sensor(modality=modality)
            def cube_to_bound_dist(obs_cache):
                if "cube_pos" in obs_cache:
                    return min(
                        np.linalg.norm(obs_cache["cube_pos"][1] - self.table_full_size[1]/2),
                        np.linalg.norm(obs_cache["cube_pos"][1] + self.table_full_size[1]/2)
                        )
                return 0
            
            @sensor(modality=modality)
            def cube_drop(obs_cache):
                if "cube_pos" in obs_cache:
                    return obs_cache["cube_pos"][2] -  self.model.mujoco_arena.table_offset[2] < 0

            sensors = [cube_pos, goal_pos, eef_to_cube_dist, cube_to_goal_dist, cube_to_bound_dist, cube_drop]
            names = [s.__name__ for s in sensors]

            for name, s in zip(names, sensors):
                observables[name] = Observable(
                    name=name,
                    sensor=s,
                    sampling_rate=2*self.control_freq,
                )
        return observables

    def _reset_internal(self):
        super()._reset_internal()
        if not self.deterministic_reset:
            placements = self.placement_initializer.sample()
            for obj_pos, obj_quat, obj in placements.values():
                self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([np.array(obj_pos), np.array(obj_quat)]))

        cube_xy = self.sim.data.body_xpos[self.cube_body_id][:2].copy()
        self.goal_xy =  cube_xy + self.goal_pos_offset
        self.sim.forward()
        cube_pos = self.sim.data.body_xpos[self.cube_body_id]
        self._last_obj_goal_dist = np.linalg.norm(cube_pos[:2] - self.goal_xy)

    def visualize(self, vis_settings):
        super().visualize(vis_settings=vis_settings)
        if vis_settings.get("grippers", False):
            self._visualize_gripper_to_target(
                gripper=self.robots[0].gripper,
                target=None,
            )
            eef_pos = self.sim.data.site_xpos[self.robots[0].eef_site_id]
            table_z = self.model.mujoco_arena.table_offset[2]
            goal_pos =  np.array([self.goal_xy[0], self.goal_xy[1], table_z + 0.001], dtype=float)
            d = np.linalg.norm(eef_pos - goal_pos)
            alpha = 1 - np.tanh(5.0 * d)
            if hasattr(self.robots[0], "gripper_visualization_sites") and self.robots[0].gripper_visualization_sites:
                for site in self.robots[0].gripper_visualization_sites:
                    site_id = self.sim.model.site_name2id(site)
                    self.sim.model.site_rgba[site_id] = np.array([0.0, 1.0, 0.0, 0.2 + 0.6 * alpha])