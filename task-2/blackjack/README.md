# Blackjack Q-learning 教程

本项目使用Q-learning算法和OpenAI Gym中的Blackjack环境来解决二十一点问题。

## 内容说明

- `blackjack_q_learning.ipynb`: Jupyter Notebook教程
- `blackjack_q_learning.py`: Python脚本

## 项目内容

本项目包含以下内容：

1. **环境介绍**：解释Blackjack环境的状态空间、动作空间和奖励机制
2. **Q-learning智能体**：实现一个简单的Q-learning算法
3. **训练过程**：训练智能体学习最优策略
4. **结果可视化**：可视化有A和无A情况下的状态价值函数和策略

## 如何运行

1. 确保已安装必要的依赖包：
   ```bash
   pip install gymnasium numpy matplotlib seaborn tqdm
   ```
2. 启动Jupyter Notebook：
   ```bash
   jupyter notebook
   ```
3. 打开`blackjack_q_learning.ipynb`文件进行运行