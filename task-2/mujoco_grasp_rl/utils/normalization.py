import gymnasium as gym
import numpy as np

class NormalizeObservationWrapper(gym.ObservationWrapper):
    """
    A wrapper that normalizes the observation space.
    It uses a running mean and variance to normalize observations to N(0, 1).
    """
    def __init__(self, env, epsilon=1e-8, clip=10.0):
        super().__init__(env)
        self.epsilon = epsilon
        self.clip = clip
        self.running_mean = np.zeros(env.observation_space.shape, dtype=np.float32)
        self.running_var = np.ones(env.observation_space.shape, dtype=np.float32)
        self.count = 1e-4

    def observation(self, observation):
        self.count += 1
        batch_mean = observation
        batch_var = 0 # Single sample batch
        
        # Update running statistics (Welford's algorithm or simple EMA)
        # Here we use simple EMA for simplicity in online learning or just simple accumulation
        # Actually standard RL uses cumulative moving average.
        
        delta = observation - self.running_mean
        self.running_mean += delta / self.count
        self.running_var += delta * (observation - self.running_mean)
        
        # Calculate std
        var = self.running_var / (self.count - 1 if self.count > 1 else 1)
        std = np.sqrt(var + self.epsilon)
        
        normalized_obs = (observation - self.running_mean) / std
        return np.clip(normalized_obs, -self.clip, self.clip)
