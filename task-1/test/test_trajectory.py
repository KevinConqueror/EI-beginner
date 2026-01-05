import numpy as np
from controllers.trajectory import TrajectoryGenerator

def test_trajectory():
    print("Testing Trajectory Generator...")
    traj_gen = TrajectoryGenerator()
    
    start_q = np.array([0, 0, 0])
    end_q = np.array([1, 2, -1])
    duration = 2.0
    dt = 0.01
    
    times, pos, vel, acc = traj_gen.generate_joint_trajectory(start_q, end_q, duration, dt)
    
    print(f"Generated {len(times)} points.")
    print(f"Start Pos: {pos[0]}")
    print(f"End Pos: {pos[-1]}")
    
    # Verify limits
    assert np.allclose(pos[0], start_q), "Start position mismatch"
    assert np.allclose(pos[-1], end_q), "End position mismatch"
    assert np.allclose(vel[0], 0), "Start velocity not zero"
    assert np.allclose(vel[-1], 0), "End velocity not zero"
    
    print("Trajectory verified successfully!")

if __name__ == "__main__":
    test_trajectory()
