# End-to-End Joint-Space Grasping with RL

This project implements a Reinforcement Learning (SAC) agent that learns to grasp a cube using a Franka Panda robot arm.
Crucially, the policy operates directly in **Joint Space**, outputting joint angle commands, without using Inverse Kinematics (IK) or scripted grasp strategies.

## Project Structure

- `assets/`: MuJoCo XML models (Franka robot, scene).
- `envs/`: Custom Gymnasium environment (`JointGraspEnv`) and Reward logic.
- `train/`: Training (`train_sac.py`) and Evaluation (`eval_policy.py`) scripts.
- `configs/`: Hyperparameters.
- `utils/`: Helper functions.

## Installation

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training

To train the SAC agent:

```bash
python train/train_sac.py
```

This will save checkpoints and logs to the `logs/` directory.

### Evaluation

To visualize a trained policy:

```bash
python train/eval_policy.py --model_path logs/sac_grasp_final.zip --norm_path logs/vec_normalize.pkl
```

## Task Details

- **Observation**: 20-dim vector (Joint angles, Joint velocities, Relative EE position, Object height, Gripper state).
- **Action**: 8-dim vector (7 Joint deltas + 1 Gripper command).
- **Reward**: Shaped reward including reach distance, contact bonus, lift bonus, and success bonus.
- **Constraints**: No IK allowed. Policy controls joints directly.

## Implementation Notes

- The environment uses `mujoco` Python bindings directly.
- `VecNormalize` from Stable Baselines 3 is used to normalize observations and rewards.
- The Franka robot is modeled with geometric primitives (capsules/cylinders) to ensure the simulation runs without external mesh dependencies.
