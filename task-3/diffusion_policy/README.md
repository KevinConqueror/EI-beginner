# Diffusion Policy for Vision-Based Robotic Tasks

## 项目概述
本项目复现**模仿学习领域经典方法**——基于扩散模型（Diffusion Model）的视觉策略学习框架，源自[官方实现](https://diffusion-policy.cs.columbia.edu/)。核心算法实现在`diffusion_policy.py`中，通过扩散过程建模高维动作空间，显著提升机器人操作任务的策略学习效果。

## 方法背景
作为模仿学习的突破性进展，Diffusion Policy通过以下创新解决传统方法痛点：
- **多模态动作预测**：克服确定性策略的单峰限制
- **轨迹优化**：直接生成完整动作序列而非单步动作
- **端到端视觉输入**：直接处理图像观测无需手工特征

## 功能特点
- 支持从图像观测中生成机器人动作序列
- 针对PUSHT（Push Task）环境优化的策略模型
- 提供Jupyter Notebook和Python脚本双版本演示


### 运行demo
```bash
# 运行Python脚本演示
python diffusion_policy_vision_pusht_demo.py

# 或启动Jupyter Notebook
jupyter notebook diffusion_policy_vision_pusht_demo.ipynb
```

## 项目结构
```
├── diffusion_policy.py          # 扩散策略核心实现（含DDPM+Transformer架构）
├── diffusion_policy_vision_pusht_demo.ipynb  # 交互式演示（含可视化轨迹）
└── diffusion_policy_vision_pusht_demo.py     # 命令行演示脚本
```

## 使用说明
1. 确保已安装所有依赖项
2. 准备符合要求的图像输入数据
3. 调整`diffusion_policy_vision_pusht_demo.py`中的参数配置
4. 执行demo脚本观察策略输出