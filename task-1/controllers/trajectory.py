import numpy as np

def quintic_polynomial_scaling(t, T):
    """
    Returns s(t), s'(t), s''(t) for quintic polynomial scaling from 0 to 1.
    s(0)=0, s(T)=1, v(0)=0, v(T)=0, a(0)=0, a(T)=0
    """
    if t < 0:
        return 0.0, 0.0, 0.0
    if t > T:
        return 1.0, 0.0, 0.0
        
    tau = t / T
    tau2 = tau * tau
    tau3 = tau2 * tau
    tau4 = tau3 * tau
    tau5 = tau4 * tau
    
    s = 10*tau3 - 15*tau4 + 6*tau5
    s_d = (30*tau2 - 60*tau3 + 30*tau4) / T
    s_dd = (60*tau - 180*tau2 + 120*tau3) / (T*T)
    
    return s, s_d, s_dd

class TrajectoryGenerator:
    def __init__(self):
        pass
        
    def generate_joint_trajectory(self, start_q, end_q, duration, dt):
        """
        Generate a joint space trajectory using quintic polynomial interpolation.
        
        Args:
            start_q: Start joint positions (numpy array)
            end_q: End joint positions (numpy array)
            duration: Total time duration
            dt: Time step
            
        Returns:
            times: Time points
            positions: Joint positions at each time step (N x n_joints)
            velocities: Joint velocities (N x n_joints)
            accelerations: Joint accelerations (N x n_joints)
        """
        start_q = np.array(start_q)
        end_q = np.array(end_q)
        
        num_steps = int(duration / dt)
        times = np.linspace(0, duration, num_steps)
        
        positions = []
        velocities = []
        accelerations = []
        
        diff = end_q - start_q
        
        for t in times:
            s, s_d, s_dd = quintic_polynomial_scaling(t, duration)
            
            q = start_q + s * diff
            dq = s_d * diff
            ddq = s_dd * diff
            
            positions.append(q)
            velocities.append(dq)
            accelerations.append(ddq)
            
        return times, np.array(positions), np.array(velocities), np.array(accelerations)
