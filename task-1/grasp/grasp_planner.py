import numpy as np
import mujoco

class GraspPlanner:
    def __init__(self, grasp_offset=0.0, pre_grasp_height=0.1):
        """
        Simple Rule-based Grasp Planner.
        
        Args:
            grasp_offset: Vertical offset from object center to grasp center.
                        Positive means higher.
            pre_grasp_height: Height above grasp pose for pre-grasp.
        """
        self.grasp_offset = grasp_offset
        self.pre_grasp_height = pre_grasp_height
        
    def plan_grasp(self, object_pos, object_quat):
        """
        Plan grasp pose for a top-down grasp.
        
        Args:
            object_pos: Object position [x, y, z]
            object_quat: Object quaternion [w, x, y, z]
            
        Returns:
            grasp_pos: Target position for end-effector
            grasp_quat: Target quaternion for end-effector
            pre_grasp_pos: Pre-grasp position
        """
        object_pos = np.array(object_pos)

        grasp_pos = object_pos.copy()
        grasp_pos[2] += self.grasp_offset

        pre_grasp_pos = grasp_pos.copy()
        pre_grasp_pos[2] += self.pre_grasp_height
        
        mat = np.array([
            [1, 0, 0],
            [0, -1, 0],
            [0, 0, -1]
        ])

        grasp_quat = np.zeros(4)
        mujoco.mju_mat2Quat(grasp_quat, mat.flatten())
        
        return grasp_pos, grasp_quat, pre_grasp_pos
