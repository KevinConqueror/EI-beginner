# EI-Beginner Implementation

本项目是 [OpenMOSS EI-Beginner 具身智能入门练习](docs/README.md) 的实现代码仓库。

## 项目简介

本项目旨在完成 OpenMOSS 实验室提供的具身智能/人形机器人智能入门练习。涵盖了从传统机器人运动学、强化学习、模仿学习到最新的 VLA 大模型和 LLM/VLM 任务规划等多个前沿领域。

所有 6 个任务均已完成，代码结构清晰，分模块管理。

## 目录结构与任务说明

本项目按照教程任务分为 6 个主要模块，每个模块包含具体的实现代码和文档。

### [Task 1: 基于传统运动学的机械臂物体抓取](./task-1)

- **路径**: `task-1/mujoco_grasp_tradition`
- **内容**: 
  - 基于 MuJoCo 仿真环境，实现了 Franka Panda 机械臂的物体抓取。
  - 包含了逆运动学 (IK) 求解器、轨迹规划 (Trajectory Planning) 和状态机控制逻辑。
  - 不依赖深度学习，纯基于几何和物理模型。
- **更多详情**: 请查阅 [Task 1 mujoco_grasp_tradition README](./task-1/mujoco_grasp_tradition/README.md)。

### [Task 2: 基于强化学习的机械臂物体抓取](./task-2)

- **路径**: 
  - `task-2/blackjack`: 强化学习基础练习 (Blackjack)。
  - `task-2/mujoco_grasp_rl`: 基于 SAC 算法的机械臂抓取训练。
- **内容**:
  - 在 MuJoCo 环境中自定义了 Gym 接口。
  - 实现了 SAC (Soft Actor-Critic) 算法，直接在关节空间控制机械臂完成抓取任务。
  - 包含了训练 (`train_sac.py`) 和评估 (`eval_policy.py`) 脚本。
- **更多详情**: 请查阅 [Task 2 balckjack README](./task-2/blackjack/README.md) 和 [Task 2 mujoco_grasp_rl README](./task-2/mujoco_grasp_rl/README.md)。

### [Task 3: 基于模仿学习的机械臂物体抓取](./task-3)

- **路径**: `task-3/diffusion_policy`
- **内容**:
  - 复现了经典的 Diffusion Policy 方法。
  - 实现了基于视觉输入的端到端动作生成。
  - 提供了 Jupyter Notebook 和 Python 脚本演示。
- **更多详情**: 请查阅 [Task 3 diffusion_policy README](./task-3/diffusion_policy/README.md)。

### [Task 4: 基于 VLA 大模型的机械臂物体抓取](./task-4)

- **路径**: `task-4/`
- **内容**:
  - `openvla`: OpenVLA 模型的推理演示。
  - `openpi`: OpenPI 模型的推理演示。
- **更多详情**: 请查阅 [Task 4 openvla README](./task-4/openvla/README.md) 和 [Task 4 openpi README](./task-4/openpi/README.md)。

### [Task 5: 基于 LLM/VLM 大模型的任务规划](./task-5)

- **路径**: 
  - `task-5/table_planning`: 桌面级任务规划 (Code as Policies)。
  - `task-5/scene_planning`: 场景级任务规划 (Habitat-Sim + Partnr-Planner)。
- **内容**:
  - **桌面级**: 利用 LLM 生成策略代码控制机械臂。
  - **场景级**: 在 Habitat 仿真环境中，利用 LLM/VLM 进行复杂的家庭环境任务规划和执行。
- **更多详情**: 请查阅 [Task 5 table_planning README](./task-5/table_planning/README.md) 和 [Task 5 scene_planning README](./task-5/scene_planning/README.md)。

### [Task 6: 基于强化学习的人形机器人运动控制](./task-6)

- **路径**: `task-6/OmniH2O`
- **内容**:
  - 复现了 OmniH2O 人形机器人运动控制方法。
  - 包含全身遥操作 (Teleoperation) 和模仿学习相关代码。
  - 基于 Isaac Gym / Legged Gym 框架。

## 使用说明

每个任务文件夹下通常都有独立的 `README.md` 和 `requirements.txt`。请进入相应的子目录查看具体的安装和运行指南。


## 参考资料

- **教程来源**: [OpenMOSS EI-Beginner](docs/README.md)
- **OpenMOSS 实验室**: https://openmoss.github.io/
