import numpy as np

class GraspReward:
    def __init__(self, config):
        self.w_reach = config.get("w_reach", 1.0)
        self.w_contact = config.get("w_contact", 1.0)
        self.w_lift = config.get("w_lift", 5.0)
        self.w_action = config.get("w_action", 0.01)
        self.w_time = config.get("w_time", 0.1)
        self.success_reward = config.get("success_reward", 100.0)
        
    def compute_reward(self, info):
        """
        Compute the reward based on the environment info.
        
        info dictionary must contain:
        - dist_ee_obj: distance between end-effector and object
        - is_contact: boolean, whether gripper is touching object
        - obj_z: current object height
        - target_z: target lift height
        - action: current action vector
        - success: boolean
        """
        dist = info['dist_ee_obj']
        is_contact = info['is_contact']
        obj_z = info['obj_z']
        target_z = info['target_z']
        action = info['action']
        success = info['success']

        # 1. Reach reward: Encourage getting close to the object
        # Use a shaped reward: 1 - tanh(10 * dist) or similar, or just negative distance
        # Logarithmic or Hyperbolic distance is often better for precision
        r_reach = -dist 
        
        # 2. Contact reward
        r_contact = 1.0 if is_contact else 0.0
        
        # 3. Lift reward: Reward for lifting the object
        # Continuous reward based on height gain
        initial_z = info.get('initial_z', 0.425) # Approx table height + half obj
        z_gain = max(0, obj_z - initial_z)
        r_lift = z_gain * 10.0 # Scale up
        
        if obj_z > target_z:
            r_lift += 1.0 # Bonus for reaching target height
            
        # 4. Action penalty (regularization)
        r_action = -np.sum(np.square(action))
        
        # 5. Success reward (sparse)
        r_success = self.success_reward if success else 0.0
        
        # 6. Time penalty
        r_time = -1.0
        
        # Total
        reward = (
            self.w_reach * r_reach +
            self.w_contact * r_contact +
            self.w_lift * r_lift +
            self.w_action * r_action +
            self.w_time * r_time +
            r_success
        )
        
        return reward, {
            "r_reach": r_reach,
            "r_contact": r_contact,
            "r_lift": r_lift,
            "r_action": r_action,
            "r_success": r_success
        }
