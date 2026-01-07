import numpy as np
import enum
import time
from controllers.ik_solver import IKSolver
from controllers.trajectory import TrajectoryGenerator
from controllers.joint_controller import JointController
from grasp.grasp_planner import GraspPlanner
import mujoco

class State(enum.Enum):
    INIT = 0
    MOVE_PRE_GRASP = 1
    DESCEND = 2
    GRASP = 3
    LIFT = 4
    DONE = 5

class GraspFSM:
    def __init__(self, model, data, joint_names):
        self.model = model
        self.data = data
        self.joint_names = joint_names
        
        # Modules
        self.ik_solver = IKSolver(model, data, joint_names, tol=1e-3)
        self.traj_gen = TrajectoryGenerator()
        self.controller = JointController(model, data, joint_names, kp=2000, kd=100)
        self.planner = GraspPlanner(grasp_offset=0.0, pre_grasp_height=0.2)
        
        # State
        self.state = State.INIT
        self.start_time = 0.0
        self.trajectory = None # (times, pos, vel, acc)
        self.traj_idx = 0
        self.settle_duration = 1.0
        
        # Targets
        self.target_q = None
        self.gripper_closed = False
        
        # Gripper control
        self.gripper_actuator_ids = []
        # In panda.xml: ctrl_finger1, ctrl_finger2, Joint names: finger_joint1, finger_joint2
        finger_names = ["finger_joint1", "finger_joint2"]
        for name in finger_names:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for i in range(model.nu):
                if model.actuator_trnid[i, 0] == jid:
                    self.gripper_actuator_ids.append(i)
                    break
                    
        print(f"Gripper Actuators: {self.gripper_actuator_ids}")
        
    def update(self, current_time):
        """
        Update FSM state and return control signals.
        """
        # Run state logic
        if self.state == State.INIT:
            self._handle_init(current_time)
        elif self.state == State.MOVE_PRE_GRASP:
            self._handle_trajectory_following(current_time, next_state=State.DESCEND)
        elif self.state == State.DESCEND:
            self._handle_trajectory_following(current_time, next_state=State.GRASP)
        elif self.state == State.GRASP:
            self._handle_grasp(current_time)
        elif self.state == State.LIFT:
            self._handle_trajectory_following(current_time, next_state=State.DONE)
        elif self.state == State.DONE:
            pass
            
        # Apply control
        if self.target_q is not None:
            target_dq = np.zeros(len(self.joint_names))
            if self.trajectory and self.state in [State.MOVE_PRE_GRASP, State.DESCEND, State.LIFT]:
                times, positions, velocities, accelerations = self.trajectory
                t = current_time - self.start_time
                dt = self.model.opt.timestep
                idx = int(t / dt)
                if idx < len(velocities):
                    target_dq = velocities[idx]
                else:
                    target_dq = np.zeros(len(self.joint_names))

            self.controller.step(self.target_q, target_dq)
        
        gripper_ctrl = 255 if self.gripper_closed else 0

        target_finger = 0.0 if self.gripper_closed else 0.04
        
        for act_id in self.gripper_actuator_ids:
            self.data.ctrl[act_id] = target_finger
            
    def _handle_init(self, current_time):
        # Wait for physics to settle
        if self.start_time == 0.0 and current_time > 0.001:
            self.start_time = current_time
        if current_time - self.start_time < self.settle_duration:
            return

        print("FSM: INIT -> MOVE_PRE_GRASP")
        
        # Get object pose
        box_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "box")
        obj_pos = self.data.xpos[box_id]
        obj_quat = self.data.xquat[box_id]
        
        print(f"Object at: {obj_pos}")
        
        # Plan grasp
        grasp_pos, grasp_quat, pre_grasp_pos = self.planner.plan_grasp(obj_pos, obj_quat)
        
        self.grasp_pos = grasp_pos
        self.grasp_quat = grasp_quat
        self.pre_grasp_pos = pre_grasp_pos
        
        # Solve IK for Grasp First
        q_current = self.data.qpos[self.controller.qpos_indices]
        q_grasp, success_g, err_g = self.ik_solver.solve(grasp_pos, grasp_quat, q_init=q_current)
        
        if not success_g:
            print("IK Failed for Grasp! Retrying with random seed...")
            q_grasp, success_g, err_g = self.ik_solver.solve(grasp_pos, grasp_quat, q_init=np.array([0, -0.785, 0, -2.356, 0, 1.571, 0.785]))

        q_pre, success_p, err_p = self.ik_solver.solve(pre_grasp_pos, grasp_quat, q_init=q_grasp)
        
        if not success_p:
            print("IK Failed for Pre-Grasp!")
        
        self.q_pre = q_pre
        self.q_grasp = q_grasp
        
        self.trajectory = self.traj_gen.generate_joint_trajectory(
            q_current, q_pre, duration=6.0, dt=self.model.opt.timestep
        )
        self.start_time = current_time
        self.state = State.MOVE_PRE_GRASP
        
    def _handle_trajectory_following(self, current_time, next_state):
        times, positions, velocities, accelerations = self.trajectory
        
        t = current_time - self.start_time
        
        if t >= times[-1]:
            self.target_q = positions[-1]
            print(f"FSM: Transition to {next_state}")
            
            if next_state == State.DESCEND:
                
                self.trajectory = self.traj_gen.generate_joint_trajectory(
                    self.q_pre, self.q_grasp, duration=4.0, dt=self.model.opt.timestep
                )
                self.start_time = current_time
                self.state = State.DESCEND
                
            elif next_state == State.GRASP:
                self.state = State.GRASP
                self.start_time = current_time
                
            elif next_state == State.DONE:
                self.state = State.DONE
                
            return

        dt = self.model.opt.timestep
        idx = int(t / dt)
        if idx >= len(positions):
            idx = len(positions) - 1
            
        self.target_q = positions[idx]
        
    def _handle_grasp(self, current_time):
        self.gripper_closed = True
        
        # Wait for grasp to settle
        if current_time - self.start_time > 1.0:
            print("FSM: GRASP -> LIFT")
            
            lift_pos = self.grasp_pos.copy()
            lift_pos[2] += 0.2
            
            q_lift, success, err = self.ik_solver.solve(lift_pos, self.grasp_quat, q_init=self.q_grasp)
            
            self.trajectory = self.traj_gen.generate_joint_trajectory(
                self.q_grasp, q_lift, duration=4.0, dt=self.model.opt.timestep
            )
            self.start_time = current_time
            self.state = State.LIFT
