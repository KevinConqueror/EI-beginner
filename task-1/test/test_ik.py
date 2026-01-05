import mujoco
import numpy as np
from controllers.ik_solver import IKSolver

def test_ik():
    print("Testing IK Solver...")
    model = mujoco.MjModel.from_xml_path('assets/scene.xml')
    data = mujoco.MjData(model)
    
    # Initialize Solver
    joint_names = [f"joint{i}" for i in range(1, 8)]
    ik = IKSolver(model, data, joint_names, site_name="attachment_site")
    
    # Define a target
    # Let's pick a point on the table
    target_pos = [0.5, 0.0, 0.4] # x, y, z
    # Orientation: Pointing down (gripper z-axis aligned with world -z)
    # Default gripper orientation might need checking. 
    # Usually for Panda, "down" means rotating 180 deg around X or Y from initial?
    # Let's just try to reach the position first.
    
    print(f"Target Pos: {target_pos}")
    
    # Run IK
    q_sol, success, err = ik.solve(target_pos)
    
    print(f"IK Success: {success}")
    print(f"Pos Error: {err}")
    
    if success:
        # Visualize result
        for i, idx in enumerate(ik.qpos_indices):
            data.qpos[idx] = q_sol[i]
            
        mujoco.mj_forward(model, data)
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
        final_pos = data.site_xpos[site_id]
        print(f"Final Pos: {final_pos}")
        
        # Simple assertion
        assert np.linalg.norm(final_pos - target_pos) < 0.01, "IK failed to reach target accurately"
        print("IK Verified!")
    else:
        print("IK Failed to converge.")

if __name__ == "__main__":
    test_ik()
