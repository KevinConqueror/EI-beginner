import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
import os
from envs.rewards import GraspReward

class JointGraspEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 20}

    def __init__(self, render_mode=None, config=None):
        super().__init__()
        
        if config is None:
            config = {}
            
        self.render_mode = render_mode
        
        # Load MuJoCo model
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, "../assets/scene.xml")
        
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        
        # Define Action Space
        # 7 arm joints + 1 gripper action
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(8,), dtype=np.float32
        )
        
        # Define Observation Space
        # q(7), qdot(7), ee_pos_rel(3), obj_z(1), gripper_state(2) = 20
        obs_dim = 7 + 7 + 3 + 1 + 2 
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        
        # Configuration
        self.max_steps = config.get("max_steps", 200)
        self.control_step = 0.05 # Control interval (seconds)
        self.sim_dt = self.model.opt.timestep # 0.002
        self.n_substeps = int(self.control_step / self.sim_dt)
        
        self.reward_calculator = GraspReward(config)
        
        # Initial state caching
        self.init_qpos = self.data.qpos.ravel().copy()
        self.init_qvel = self.data.qvel.ravel().copy()
        
        # Identify IDs - UPDATED FOR NEW ASSETS
        self.ee_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
        self.obj_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "box")
        
        # Object geom might not have a name, so we get it from body
        # Assuming the box body has only one geom or the first one is the box
        self.obj_geom_id = self.model.body_geomadr[self.obj_body_id]
        
        # Joint IDs - UPDATED NAMES
        self.joint_ids = []
        for i in range(1, 8):
            self.joint_ids.append(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i}"))
        self.finger_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "finger_joint1"),
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "finger_joint2")
        ]
        
        # Actuator IDs - UPDATED NAMES
        self.actuator_ids = []
        for i in range(1, 8):
            self.actuator_ids.append(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"ctrl_joint{i}"))
        self.gripper_act_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "ctrl_finger1"),
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "ctrl_finger2")
        ]
        
        # Viewer
        self.viewer = None
        self._step_count = 0
        
        # Check IDs to ensure they were found (non-negative)
        if self.ee_site_id < 0: raise ValueError("EE Site not found")
        if self.obj_body_id < 0: raise ValueError("Object Body not found")
        for jid in self.joint_ids:
            if jid < 0: raise ValueError("Joint not found")

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        mujoco.mj_resetData(self.model, self.data)
        
        # Randomize object position slightly
        # Object is a free joint.
        # Find the QPOS address for the object free joint.
        # The body is "box". The joint is usually named "box_joint" or just the first joint of that body.
        # In new XML: <body name="box"><freejoint/>...
        # The joint name is likely default or we find it via body.
        # Actually, let's find the joint address from body.
        obj_jnt_adr = self.model.body_jntadr[self.obj_body_id]
        obj_qpos_adr = self.model.jnt_qposadr[obj_jnt_adr]
        
        # Initial pos: 0.5, 0, 0.5 (from XML)
        # Randomize X, Y
        self.data.qpos[obj_qpos_adr] = 0.5 + np.random.uniform(-0.05, 0.05)
        self.data.qpos[obj_qpos_adr+1] = 0.0 + np.random.uniform(-0.05, 0.05)
        self.data.qpos[obj_qpos_adr+2] = 0.45 # Drop it slightly above table (table top ~0.42)
        
        # Reset arm to neutral
        # Neutral pose for Franka:
        neutral_q = np.array([0, -0.785, 0, -2.356, 0, 1.571, 0.785])
        for i, jid in enumerate(self.joint_ids):
             self.data.qpos[self.model.jnt_qposadr[jid]] = neutral_q[i]
             
        mujoco.mj_forward(self.model, self.data)
        
        self._step_count = 0
        self.initial_obj_z = self.data.xpos[self.obj_body_id][2]
        
        return self._get_obs(), {}

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        
        # Split action
        arm_action = action[:7]
        gripper_action = action[7]
        
        # Apply control (Delta Position Control)
        delta_scale = 0.05 
        current_q = np.array([self.data.qpos[self.model.jnt_qposadr[jid]] for jid in self.joint_ids])
        target_q = current_q + arm_action * delta_scale
        
        for i, jid in enumerate(self.joint_ids):
             min_limit = self.model.jnt_range[jid][0]
             max_limit = self.model.jnt_range[jid][1]
             target_q[i] = np.clip(target_q[i], min_limit, max_limit)
             self.data.ctrl[self.actuator_ids[i]] = target_q[i]
             
        # Gripper control
        # New XML range: 0 to 0.04
        # Action -1 (open) -> 0.04, 1 (close) -> 0
        gripper_target = 0.02 * (1 - gripper_action) 
        
        for aid in self.gripper_act_ids:
            self.data.ctrl[aid] = gripper_target
            
        # Step simulation
        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)
            
        self._step_count += 1
        
        obs = self._get_obs()
        info = self._get_info(action)
        reward, r_info = self.reward_calculator.compute_reward(info)
        info.update(r_info)
        
        terminated = info['success'] or (self._step_count >= self.max_steps)
        truncated = False
        
        if not np.all(np.isfinite(obs)):
            terminated = True
            reward = -10.0
            
        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        # 1. Joint angles (7)
        q = np.array([self.data.qpos[self.model.jnt_qposadr[jid]] for jid in self.joint_ids])
        
        # 2. Joint velocities (7)
        qdot = np.array([self.data.qvel[self.model.jnt_dofadr[jid]] for jid in self.joint_ids])
        
        # 3. EE pos relative to object (3)
        ee_pos = self.data.site_xpos[self.ee_site_id]
        obj_pos = self.data.xpos[self.obj_body_id]
        rel_pos = ee_pos - obj_pos
        
        # 4. Object height (1)
        obj_z = np.array([obj_pos[2]])
        
        # 5. Gripper state (2)
        g1 = self.data.qpos[self.model.jnt_qposadr[self.finger_ids[0]]]
        g2 = self.data.qpos[self.model.jnt_qposadr[self.finger_ids[1]]]
        gripper_state = np.array([g1, g2])
        
        return np.concatenate([q, qdot, rel_pos, obj_z, gripper_state]).astype(np.float32)

    def _get_info(self, action):
        ee_pos = self.data.site_xpos[self.ee_site_id]
        obj_pos = self.data.xpos[self.obj_body_id]
        dist_ee_obj = np.linalg.norm(ee_pos - obj_pos)
        
        # Check contact
        is_contact = False
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            g1 = contact.geom1
            g2 = contact.geom2
            
            # Check if one of them is the object geom
            c1 = (g1 == self.obj_geom_id)
            c2 = (g2 == self.obj_geom_id)
            
            if c1 or c2:
                is_contact = True
                break
        
        obj_z = obj_pos[2]
        target_z = self.initial_obj_z + 0.1
        
        success = (obj_z > target_z) and (dist_ee_obj < 0.05)
        
        return {
            "dist_ee_obj": dist_ee_obj,
            "is_contact": is_contact,
            "obj_z": obj_z,
            "target_z": target_z,
            "initial_z": self.initial_obj_z,
            "action": action,
            "success": success
        }

    def render(self):
        if self.render_mode == "human":
            if self.viewer is None:
                from mujoco import viewer
                self.viewer = viewer.launch_passive(self.model, self.data)
            self.viewer.sync()
        elif self.render_mode == "rgb_array":
            renderer = mujoco.Renderer(self.model)
            renderer.update_scene(self.data)
            return renderer.render()
            
    def close(self):
        if self.viewer is not None:
            self.viewer.close()
