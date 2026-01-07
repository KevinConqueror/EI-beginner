import mujoco
import numpy as np

class JointController:
    def __init__(self, model, data, joint_names, kp=100, kd=20):
        """
        Joint Impedance Controller (PD + Gravity Compensation).
        
        Args:
            model: Mujoco model
            data: Mujoco data
            joint_names: List of joint names to control
            kp: Proportional gain
            kd: Derivative gain
        """
        self.model = model
        self.data = data
        self.joint_names = joint_names
        self.kp = kp
        self.kd = kd
        
        # Map joints to actuators and dof indices
        self.actuator_ids = []
        self.dof_indices = []
        self.qpos_indices = []
        
        for name in joint_names:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid == -1:
                raise ValueError(f"Joint '{name}' not found")
            
            found = False
            for i in range(model.nu):
                if model.actuator_trnid[i, 0] == jid:
                    self.actuator_ids.append(i)
                    found = True
                    break
            if not found:
                print(f"Warning: No actuator found for joint {name}")
                self.actuator_ids.append(-1)
            else:
                pass
                
            self.qpos_indices.append(model.jnt_qposadr[jid])
            self.dof_indices.append(model.jnt_dofadr[jid])
            
        self.actuator_ids = np.array(self.actuator_ids, dtype=int)
        self.qpos_indices = np.array(self.qpos_indices, dtype=int)
        self.dof_indices = np.array(self.dof_indices, dtype=int)
        self.n_joints = len(joint_names)
        
    def step(self, target_q, target_dq=None):
        """
        Compute torques and apply to actuators.
        
        Args:
            target_q: Target joint positions (numpy array)
            target_dq: Target joint velocities (numpy array, optional)
        """
        if target_dq is None:
            target_dq = np.zeros(self.n_joints)
            
        current_q = self.data.qpos[self.qpos_indices]
        current_dq = self.data.qvel[self.dof_indices]
        
        # PD Control
        error_q = target_q - current_q
        error_dq = target_dq - current_dq
        
        # Torque = Kp * e + Kd * de
        bias_forces = self.data.qfrc_bias[self.dof_indices]
        gears = self.model.actuator_gear[self.actuator_ids, 0]
        gears[gears == 0] = 1.0
        desired_torque = self.kp * error_q + self.kd * error_dq + bias_forces
        ctrl_input = desired_torque / gears
        
        for i, act_id in enumerate(self.actuator_ids):
            if act_id != -1:
                self.data.ctrl[act_id] = ctrl_input[i]

