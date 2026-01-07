import os
import time
import argparse
import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from envs.joint_grasp_env import JointGraspEnv

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to saved model zip")
    parser.add_argument("--norm_path", type=str, default=None, help="Path to saved VecNormalize pkl (optional)")
    args = parser.parse_args()
    
    # Create Env with render mode "human" for visualization
    # This instructs the env to initialize the viewer
    env = JointGraspEnv(render_mode="human")
    
    # Wrap in DummyVecEnv
    # We must pass the env instance wrapped in a lambda or just the env itself if already instantiated.
    # But DummyVecEnv expects a list of constructors.
    # However, if we construct it outside, we can't easily pass it to DummyVecEnv correctly without re-initialization 
    # unless we use a custom wrapper or just use the env directly without VecEnv for evaluation.
    # SB3 models expect VecEnv for prediction usually, but can handle gym.Env.
    # But for VecNormalize, we need the VecEnv wrapper.
    
    # Correct way to wrap existing env in DummyVecEnv:
    # env = DummyVecEnv([lambda: env])  <- This fails because lambda captures the *initialized* env, but DummyVecEnv calls reset() which is fine.
    # However, standard practice is to pass a constructor.
    # Let's assume we want to use the same env instance to keep the viewer alive.
    
    # Re-creating env inside lambda is better, but we need to pass render_mode="human"
    env = DummyVecEnv([lambda: JointGraspEnv(render_mode="human")])
    
    # Load normalization stats if provided
    if args.norm_path:
        print(f"Loading normalization stats from {args.norm_path}")
        env = VecNormalize.load(args.norm_path, env)
        env.training = False # Do not update stats at test time
        env.norm_reward = False
    
    # Load Model
    print(f"Loading model from {args.model_path}")
    model = SAC.load(args.model_path)
    
    obs = env.reset()
    print("Starting evaluation... Press Ctrl+C to stop.")
    
    try:
        while True:
            # Predict action
            action, _states = model.predict(obs, deterministic=True)
            
            # Step environment
            obs, rewards, dones, infos = env.step(action)
            
            # Render
            # Since we wrapped it in DummyVecEnv, we should call env.render() on the VecEnv
            env.render()
            
            # Slow down for visualization
            time.sleep(0.02) # 50Hz
            
            if dones[0]:
                print(f"Episode finished. Reward: {rewards[0]}")
                # VecEnv resets automatically
                
    except KeyboardInterrupt:
        print("Evaluation stopped.")
        env.close()

if __name__ == "__main__":
    main()
