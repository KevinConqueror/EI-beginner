import mujoco
import numpy as np

class IKSolver:
    def __init__(self, model, data, joint_names, site_name="attachment_site", damping=1e-2, step_size=0.5, max_steps=100, tol=1e-3):
        """
        Numerical IK Solver using Jacobian Damped Least Squares.
        
        Args:
            model: Mujoco model
            data: Mujoco data
            joint_names: List of joint names to control
            site_name: Name of the end-effector site
            damping: Damping factor for DLS (lambda)
            step_size: Step size for integration
            max_steps: Maximum iterations
            tol: Convergence tolerance (position error)
        """
        self.model = model
        self.data = data
        self.site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if self.site_id == -1:
            raise ValueError(f"Site '{site_name}' not found.")
            
        self.damping = damping
        self.step_size = step_size
        self.max_steps = max_steps
        self.tol = tol
        
        # Identify joint indices
        self.qpos_indices = []
        self.dof_indices = []
        
        for name in joint_names:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid == -1:
                raise ValueError(f"Joint '{name}' not found.")
            
            q_adr = model.jnt_qposadr[jid]
            dof_adr = model.jnt_dofadr[jid]
            
            # Assuming 1-DOF joints (hinge/slide) for now
            self.qpos_indices.append(q_adr)
            self.dof_indices.append(dof_adr)
            
        self.qpos_indices = np.array(self.qpos_indices, dtype=int)
        self.dof_indices = np.array(self.dof_indices, dtype=int)
        self.n_joints = len(self.qpos_indices)
        
        # Cache for full Jacobian
        self.nv = model.nv
        self.jac = np.zeros((6, self.nv))
        self.jac_pos = self.jac[:3]
        self.jac_rot = self.jac[3:]
        
    def solve(self, target_pos, target_quat=None, q_init=None):
        """
        Solve IK for target position and quaternion.
        
        Args:
            target_pos: Target position [x, y, z]
            target_quat: Target orientation [w, x, y, z] (optional)
            q_init: Initial joint configuration (optional, uses current data.qpos if None)
            
        Returns:
            q_solution: Solution joint angles (numpy array corresponding to joint_names)
            success: Boolean indicating convergence
            err_norm: Final position error norm
        """
        if q_init is not None:
            if len(q_init) != self.n_joints:
                raise ValueError(f"q_init size {len(q_init)} != n_joints {self.n_joints}")
            self.data.qpos[self.qpos_indices] = q_init
            mujoco.mj_forward(self.model, self.data)
            
        target_pos = np.array(target_pos)
        solve_rot = target_quat is not None
        
        # We work with a local copy of joint angles
        current_q = self.data.qpos[self.qpos_indices].copy()
        
        for i in range(self.max_steps):
            # 1. Update Simulation
            self.data.qpos[self.qpos_indices] = current_q
            mujoco.mj_forward(self.model, self.data)
            
            # 2. Get current end-effector pose
            current_pos = self.data.site_xpos[self.site_id]
            current_mat = self.data.site_xmat[self.site_id].reshape(3, 3)
            
            # 3. Compute Error
            err_pos = target_pos - current_pos
            
            error = np.zeros(6)
            error[:3] = err_pos
            
            if solve_rot:
                # Orientation error
                current_quat = np.zeros(4)
                mujoco.mju_mat2Quat(current_quat, current_mat.flatten())
                
                neg_current_quat = np.zeros(4)
                mujoco.mju_negQuat(neg_current_quat, current_quat)
                
                diff_quat = np.zeros(4)
                mujoco.mju_mulQuat(diff_quat, target_quat, neg_current_quat)
                
                w = diff_quat[0]
                vec = diff_quat[1:]
                
                if w < 0:
                    w = -w
                    vec = -vec
                    
                err_rot = 2.0 * vec # Small angle approx
                error[3:] = err_rot
            
            # Check convergence
            if np.linalg.norm(error[:3]) < self.tol:
                if not solve_rot or np.linalg.norm(error[3:]) < 0.1:
                    return current_q, True, np.linalg.norm(error[:3])
            
            # 4. Compute Jacobian
            mujoco.mj_jacSite(self.model, self.data, self.jac_pos, self.jac_rot, self.site_id)
            
            # Extract only columns corresponding to controlled joints
            J_full = self.jac if solve_rot else self.jac_pos
            J = J_full[:, self.dof_indices]
            
            # 5. Solve DLS
            lambda_sq = self.damping ** 2
            n_task = 6 if solve_rot else 3
            
            # JJT is n_task x n_task
            JJT = J @ J.T
            damped_inv = np.linalg.inv(JJT + lambda_sq * np.eye(n_task))
            
            dq = J.T @ damped_inv @ (error if solve_rot else error[:3])
            
            # 6. Update q
            current_q += self.step_size * dq
            
        return current_q, False, np.linalg.norm(error[:3])
