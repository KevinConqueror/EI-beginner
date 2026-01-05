# MuJoCo Franka Panda 抓取demo

本项目使用 MuJoCo Python 绑定实现了一个传统的机器人抓取流程。
它控制 Franka Panda 机器人从桌子上抓取一个立方体。

## 特性

- **MuJoCo 仿真**：基于物理的 Franka Panda 模型仿真。
- **数值逆运动学**：基于雅可比矩阵的阻尼最小二乘逆运动学。
- **轨迹规划**：使用五次多项式插值实现平滑关节运动。
- **关节控制**：阻抗控制（PD + 重力补偿）。
- **无学习**：纯基于模型和几何的方法。

## 结构

- `assets/`：MJCF 模型文件。
- `controllers/`：逆运动学、轨迹和关节控制模块。
- `grasp/`：抓取规划和状态机。
- `main.py`：主程序。

## 安装

```bash
pip install -r requirements.txt
```

## 使用方法

运行演示：

```bash
python main.py
```

查看器将打开，机器人将：
1. 移动到立方体上方的预抓取位置。
2. 下降到抓取位置。
3. 闭合夹爪。
4. 提起立方体。

## 模块

1.  **场景**：`assets/scene.xml` 定义了环境。
2.  **IK 求解器**：`controllers/ik_solver.py` 解决逆运动学问题。
3.  **轨迹**：`controllers/trajectory.py` 生成平滑关节路径。
4.  **控制器**：`controllers/joint_controller.py` 计算关节力矩。
5.  **抓取规划器**：`grasp/grasp_planner.py` 计算抓取目标。
6.  **FSM**：`grasp/state_machine.py` 管理工作流程。