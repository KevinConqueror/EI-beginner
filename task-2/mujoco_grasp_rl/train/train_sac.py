import os
import yaml
import argparse
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

# Add project root to path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from envs.joint_grasp_env import JointGraspEnv
from utils.logging import setup_logger

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    logger = setup_logger()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/sac.yaml", help="Path to config file")
    parser.add_argument("--log_dir", type=str, default="logs", help="Log directory")
    args = parser.parse_args()
    
    # Load Config
    config_path = os.path.join(os.path.dirname(__file__), "..", args.config)
    config = load_config(config_path)
    logger.info(f"Loaded config from {config_path}")
    
    # Create Env
    def make_env():
        env = JointGraspEnv(config=config.get("env_config", {}))
        env = Monitor(env) # For logging
        return env
        
    # We use DummyVecEnv for SB3
    env = DummyVecEnv([make_env])
    
    # Normalize Observations and Rewards
    # Using SB3's VecNormalize is robust
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)
    
    # Initialize Agent
    model = SAC(
        config["policy_type"],
        env,
        learning_rate=config["learning_rate"],
        buffer_size=config["buffer_size"],
        learning_starts=config["learning_starts"],
        batch_size=config["batch_size"],
        tau=config["tau"],
        gamma=config["gamma"],
        train_freq=config["train_freq"],
        gradient_steps=config["gradient_steps"],
        ent_coef=config["ent_coef"],
        target_update_interval=config["target_update_interval"],
        verbose=1,
        tensorboard_log=args.log_dir
    )
    
    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=os.path.join(args.log_dir, "checkpoints"),
        name_prefix="sac_grasp"
    )
    
    logger.info("Starting training...")
    try:
        model.learn(
            total_timesteps=config["total_timesteps"],
            callback=checkpoint_callback,
            progress_bar=True
        )
    except KeyboardInterrupt:
        logger.info("Training interrupted by user.")
    
    # Save final model
    save_path = os.path.join(args.log_dir, "sac_grasp_final")
    model.save(save_path)
    env.save(os.path.join(args.log_dir, "vec_normalize.pkl"))
    logger.info(f"Model saved to {save_path}")

if __name__ == "__main__":
    main()
