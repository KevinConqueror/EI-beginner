import mujoco
import mujoco.viewer
import numpy as np
import time
from grasp.state_machine import GraspFSM

def main():
    print("Initializing MuJoCo...")
    model = mujoco.MjModel.from_xml_path('assets/scene.xml')
    data = mujoco.MjData(model)
    
    # Panda Neutral Pose
    q_init = np.array([0, -0.785, 0, -2.356, 0, 1.571, 0.785])
    
    joint_names = [f"joint{i}" for i in range(1, 8)]
    
    qpos_indices = []
    for name in joint_names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        qpos_indices.append(model.jnt_qposadr[jid])
        
    data.qpos[qpos_indices] = q_init
    
    mujoco.mj_forward(model, data)
    
    fsm = GraspFSM(model, data, joint_names)
    
    print("Starting Simulation...")
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        start_time = time.time()
        
        while viewer.is_running():
            step_start = time.time()
            
            # Simulation time
            sim_time = data.time
            
            # Update FSM
            fsm.update(sim_time)
            
            # Step Physics
            mujoco.mj_step(model, data)
            
            # Sync Viewer
            viewer.sync()
            
            # Real-time sync
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
                
if __name__ == "__main__":
    main()
