import mujoco
import numpy as np
import time
from controllers.joint_controller import JointController

def test_controller():
    print("Testing Joint Controller...")
    model = mujoco.MjModel.from_xml_path('assets/scene.xml')
    data = mujoco.MjData(model)
    
    joint_names = [f"joint{i}" for i in range(1, 8)]
    # Kp/Kd might need tuning. 
    # With gear=100, torques are high.
    # Try Kp=2000, Kd=100
    controller = JointController(model, data, joint_names, kp=2000, kd=100)
    
    # Initialize to a valid pose (Joint 4 range is negative)
    q_init = np.array([0, 0, 0, -1.5, 0, 0, 0])
    data.qpos[controller.qpos_indices] = q_init
    mujoco.mj_forward(model, data)
    
    # Target: Move joint 4 to -1.0
    target_q = np.array([0, 0, 0, -1.0, 0, 0, 0])
    
    print(f"Target Q: {target_q}")
    
    # Run simulation
    duration = 2.0
    dt = model.opt.timestep
    steps = int(duration / dt)
    
    print(f"Simulating for {duration} seconds...")
    
    # For visualization (optional, we can just check error)
    errors = []
    
    for _ in range(steps):
        controller.step(target_q)
        mujoco.mj_step(model, data)
        
        current_q = data.qpos[controller.qpos_indices]
        errors.append(np.linalg.norm(target_q - current_q))
        
    final_error = errors[-1]
    print(f"Final Error Norm: {final_error}")
    
    assert final_error < 0.1, "Controller failed to converge"
    print("Controller Verified!")

if __name__ == "__main__":
    test_controller()
