import numpy as np
import mujoco
from grasp.grasp_planner import GraspPlanner

def test_grasp_planner():
    print("Testing Grasp Planner...")
    planner = GraspPlanner(grasp_offset=0.02, pre_grasp_height=0.15)
    
    obj_pos = [0.5, 0.0, 0.025]
    obj_quat = [1, 0, 0, 0]
    
    g_pos, g_quat, pre_pos = planner.plan_grasp(obj_pos, obj_quat)
    
    print(f"Object Pos: {obj_pos}")
    print(f"Grasp Pos: {g_pos}")
    print(f"Pre-Grasp Pos: {pre_pos}")
    print(f"Grasp Quat: {g_quat}")
    
    # Verification
    assert np.allclose(g_pos[:2], obj_pos[:2]), "XY alignment failed"
    assert g_pos[2] > obj_pos[2], "Grasp should be above object center"
    assert pre_pos[2] > g_pos[2], "Pre-grasp should be above grasp"
    
    # Verify orientation (Z axis should be [0, 0, -1])
    mat = np.zeros(9)
    mujoco.mju_quat2Mat(mat, g_quat)
    mat = mat.reshape(3, 3)
    z_axis = mat[:, 2]
    print(f"Grasp Z-axis: {z_axis}")
    
    assert np.allclose(z_axis, [0, 0, -1]), "Orientation is not top-down"
    
    print("Grasp Planner Verified!")

if __name__ == "__main__":
    test_grasp_planner()
