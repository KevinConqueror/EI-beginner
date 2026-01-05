import mujoco
import mujoco.viewer
import time

def verify_scene():
    model = mujoco.MjModel.from_xml_path('assets/scene.xml')
    data = mujoco.MjData(model)
    
    print("Scene loaded successfully!")
    print(f"Model nbody: {model.nbody}")
    print(f"Model njnt: {model.njnt}")
    
    # Check for robot base
    try:
        link0_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link0")
        print(f"Robot base found: ID {link0_id}")
    except Exception as e:
        print(f"Robot base NOT found: {e}")
    
    # Check if we can step the simulation
    mujoco.mj_step(model, data)
    print("Simulation step successful!")
    
    # Launch viewer for manual verification
    print("Launching viewer for 5 seconds...")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        start = time.time()
        while time.time() - start < 5:
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.01)

if __name__ == "__main__":
    verify_scene()
