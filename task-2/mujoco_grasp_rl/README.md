# 端到端关节空间抓取强化学习

本项目实现了一个强化学习（SAC）智能体，用于学习使用 Franka Panda 机械臂抓取立方体。  
**关键点**：该策略直接在**关节空间**中运行，输出关节角度指令，**无需使用逆运动学（IK）或脚本化抓取策略**。

## 项目结构

- `assets/`：MuJoCo XML 模型（Franka 机器人、场景）。
- `envs/`：自定义 Gymnasium 环境（[JointGraspEnv](.\envs\joint_grasp_env.py)）及奖励逻辑。
- `train/`：训练（[train_sac.py](.\train\train_sac.py)）与评估（[eval_policy.py](.\train\eval_policy.py)）脚本。
- `configs/`：超参数配置。
- `utils/`：辅助函数。

## 安装

1. 安装依赖项：
   ```bash
   pip install -r requirements.txt
   ```

## 使用方法

### 训练

训练 SAC 智能体：
```bash
python train/train_sac.py
```
训练结果（检查点和日志）将保存至 `logs/` 目录。

### 评估

可视化已训练策略：
```bash
python train/eval_policy.py --model_path logs/sac_grasp_final.zip --norm_path logs/vec_normalize.pkl
```

## 任务细节

- **观测**：20 维向量（关节角度、关节速度、末端执行器相对位置、物体高度、夹爪状态）。
- **动作**：8 维向量（7 个关节增量 + 1 个夹爪指令）。
- **奖励**：包含到达距离、接触奖励、提升奖励和成功奖励的组合奖励。
- **约束**：禁止使用 IK。策略直接控制关节。

## 实现说明

- 环境直接使用 `mujoco` Python 绑定。
- 使用 Stable Baselines 3 的 `VecNormalize` 对观测和奖励进行归一化。
- Franka 机器人通过几何图元（胶囊体/圆柱体）建模，确保仿真无需外部网格依赖即可运行。