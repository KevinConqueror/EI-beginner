"""
使用表格Q学习解决21点游戏
==================================

本教程使用表格Q学习训练一个21点游戏的智能体。
"""

# %%
# .. image:: /_static/img/tutorials/blackjack_AE_loop.jpg
#   :width: 650
#   :alt: 代理-环境示意图
#   :class: only-light
# .. image::  /_static/img/tutorials/blackjack_AE_loop_dark.png
#   :width: 650
#   :alt: 代理-环境示意图
#   :class: only-dark
#
# 在本教程中，我们将探索并解决*Blackjack-v1*环境。
#
# **21点**是最受欢迎的赌场纸牌游戏之一，也因在特定条件下可被击败而闻名。本版本游戏使用无限牌组（抽牌后放回），
# 因此在我们的模拟游戏中无法使用算牌策略。
# 完整文档请访问 https://gymnasium.farama.org/environments/toy_text/blackjack
#
# **目标**：获胜条件是玩家的牌面总和大于庄家且不超过21点。
#
# **动作**：智能体可以选择两种动作：
#  - 停牌 (0)：玩家不再要牌
#  - 要牌 (1)：玩家将获得另一张牌，但可能超过21点爆牌
#
# **方法**：要自己解决此环境，您可以选择最喜欢的离散强化学习算法。本教程提供的解决方案使用*Q学习*
# （一种无模型强化学习算法）。
#


# %%
# 导入和环境设置
# ------------------------------
#

# 作者：Till Zemann
# 许可证：MIT许可证

from __future__ import annotations

from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Patch
from tqdm import tqdm

import gymnasium as gym


# 首先创建21点环境。
# 注意：我们将遵循Sutton & Barto书中的规则。
# 其他版本的游戏规则如下供您实验。

env = gym.make("Blackjack-v1", sab=True)

# %%
# .. code:: py
#
#   # 其他可能的环境配置：
#
#   env = gym.make('Blackjack-v1', natural=True, sab=False)
#   # 是否对自然21点（即开局就是A和10，总和21）给予额外奖励。
#
#   env = gym.make('Blackjack-v1', natural=False, sab=False)
#   # 是否遵循Sutton和Barto书中描述的精确规则。如果`sab`为`True`，则`natural`参数将被忽略。
#


# %%
# 观察环境
# ------------------------------
#
# 首先，我们调用``env.reset()``开始一个回合。此函数将环境重置到起始位置并返回初始
# ``观测值``。通常我们也会设置``done = False``。此变量稍后用于检查游戏是否结束
# （即玩家获胜或失败）。
#

# 重置环境获取初始观测值
done = False
observation, info = env.reset()

# observation = (16, 9, False)


# %%
# 注意我们的观测值是一个包含3个值的三元组：
#
# -  玩家当前总和
# -  庄家明牌的点数
# -  布尔值表示玩家是否持有可用的A（A可用是指A计为11点不会爆牌）
#


# %%
# 执行动作
# ------------------------------
#
# 收到初始观测值后，我们将仅使用``env.step(action)``函数与环境交互。此
# 函数接收一个动作作为输入并在环境中执行它。由于该动作会改变环境状态，它会返回
# 四个对我们有用的变量：
#
# -  ``next_state``: 智能体执行动作后将收到的观测值。
# -  ``reward``: 智能体执行动作后将收到的奖励。
# -  ``terminated``: 布尔变量，指示环境是否已终止。
# -  ``truncated``: 布尔变量，指示回合是否因提前截断而结束（即达到时间限制）。
# -  ``info``: 可能包含环境额外信息的字典。
#
# ``next_state``、``reward``、``terminated``和``truncated``变量不言自明，但``info``变量需要
# 额外说明。此变量包含一个字典，可能有环境的额外信息，但在Blackjack-v1
# 环境中可以忽略它。例如在Atari环境中，info字典有``ale.lives``键，告诉我们
# 智能体还剩多少条命。如果智能体生命为0，则回合结束。
#
# 注意在训练循环中调用``env.render()``不是个好主意，因为渲染会大大减慢训练速度。
# 更好的做法是构建额外的循环来评估和展示训练后的智能体。
#

# 从所有有效动作中随机采样一个动作
action = env.action_space.sample()
# action=1

# 在环境中执行动作并接收环境信息
observation, reward, terminated, truncated, info = env.step(action)

# observation=(24, 10, False)
# reward=-1.0
# terminated=True
# truncated=False
# info={}


# %%
# 一旦``terminated = True``或``truncated=True``，我们应该停止当前回合并
# 通过``env.reset()``开始新回合。如果在未重置环境的情况下继续执行动作，环境仍会响应，
# 但输出对训练无用（甚至可能因智能体学习无效数据而有害）。
#


# %%
# 构建智能体
# ------------------------------
#
# 让我们构建一个``Q-learning智能体``来解决*Blackjack-v1*！我们需要一些函数来选择动作和更新
# 智能体的动作值。为确保智能体探索环境，一种可能的解决方案是``epsilon-greedy``策略，
# 其中我们以``epsilon``概率选择随机动作，以``1 - epsilon``概率选择当前认为最佳的动作。
#

class BlackjackAgent:
    def __init__(
        self,
        env,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.95,
    ):
        """初始化一个强化学习智能体，包含空的状态-动作值字典（q_values）、学习率和epsilon。

        参数:
            learning_rate: 学习率
            initial_epsilon: 初始epsilon值
            epsilon_decay: epsilon衰减率
            final_epsilon: 最终epsilon值
            discount_factor: 用于计算Q值的折扣因子
        """
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))

        self.lr = learning_rate
        self.discount_factor = discount_factor

        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.training_error = []

    def get_action(self, env, obs: tuple[int, int, bool]) -> int:
        """
        以概率(1 - epsilon)返回最佳动作，
        否则以概率epsilon返回随机动作以确保探索环境。
        """
        # 以概率epsilon返回随机动作以探索环境
        if np.random.random() < self.epsilon:
            return env.action_space.sample()

        # 以概率(1 - epsilon)贪婪行动（利用）
        else:
            return int(np.argmax(self.q_values[obs]))

    def update(
        self,
        obs: tuple[int, int, bool],
        action: int,
        reward: float,
        terminated: bool,
        next_obs: tuple[int, int, bool],
    ):
        """更新动作的Q值。"""
        future_q_value = (not terminated) * np.max(self.q_values[next_obs])
        temporal_difference = (
            reward + self.discount_factor * future_q_value - self.q_values[obs][action]
        )

        self.q_values[obs][action] = (
            self.q_values[obs][action] + self.lr * temporal_difference
        )
        self.training_error.append(temporal_difference)

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)


# %%
# 要训练智能体，我们将让智能体一次玩一个回合（一个完整的游戏称为一个回合），然后在
# 每个步骤（游戏中的一次单动作称为一个步骤）后更新其Q值。
#
# 智能体需要经历大量回合才能充分探索环境。
#
# 现在我们应该可以构建训练循环了。
#

# 超参数
learning_rate = 0.01
n_episodes = 100000
start_epsilon = 1.0
epsilon_decay = start_epsilon / (n_episodes / 2)  # 随时间减少探索
final_epsilon = 0.1

agent = BlackjackAgent(
    env=env,
    learning_rate=learning_rate,
    initial_epsilon=start_epsilon,
    epsilon_decay=epsilon_decay,
    final_epsilon=final_epsilon,
)

# %%
# 太好了，开始训练！
#
# 说明：当前超参数设置用于快速训练一个不错的智能体。
# 如果想收敛到最优策略，尝试将n_episodes增加10倍并将learning_rate降低（例如0.001）。
#


env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)
for episode in tqdm(range(n_episodes)):
    obs, info = env.reset()
    done = False

    # 玩一个回合
    while not done:
        action = agent.get_action(env, obs)
        next_obs, reward, terminated, truncated, info = env.step(action)

        # 更新智能体
        agent.update(obs, action, reward, terminated, next_obs)

        # 更新环境是否结束和当前观测值
        done = terminated or truncated
        obs = next_obs

    agent.decay_epsilon()


# %%
# 可视化训练过程
# ------------------------------
#

rolling_length = 500
fig, axs = plt.subplots(ncols=3, figsize=(12, 5))
axs[0].set_title("回合奖励")
# 计算并分配数据的滚动平均值以提供更平滑的图表
reward_moving_average = (
    np.convolve(
        np.array(env.return_queue).flatten(), np.ones(rolling_length), mode="valid"
    )
    / rolling_length
)
axs[0].plot(range(len(reward_moving_average)), reward_moving_average)
axs[1].set_title("回合长度")
length_moving_average = (
    np.convolve(
        np.array(env.length_queue).flatten(), np.ones(rolling_length), mode="same"
    )
    / rolling_length
)
axs[1].plot(range(len(length_moving_average)), length_moving_average)
axs[2].set_title("训练误差")
training_error_moving_average = (
    np.convolve(np.array(agent.training_error), np.ones(rolling_length), mode="same")
    / rolling_length
)
axs[2].plot(range(len(training_error_moving_average)), training_error_moving_average)
plt.tight_layout()
plt.show()

# %%
# .. image:: /_static/img/tutorials/blackjack_training_plots.png
#


# %%
# 可视化策略
# ------------------------------


def create_grids(agent, usable_ace=False):
    """根据智能体创建值和策略网格。"""
    # 将状态-动作值转换为状态值
    # 并构建将观测映射到动作的策略字典
    state_value = defaultdict(float)
    policy = defaultdict(int)
    for obs, action_values in agent.q_values.items():
        state_value[obs] = float(np.max(action_values))
        policy[obs] = int(np.argmax(action_values))

    player_count, dealer_count = np.meshgrid(
        # 玩家点数，庄家明牌
        np.arange(12, 22),
        np.arange(1, 11),
    )

    # 为绘图创建值网格
    value = np.apply_along_axis(
        lambda obs: state_value[(obs[0], obs[1], usable_ace)],
        axis=2,
        arr=np.dstack([player_count, dealer_count]),
    )
    value_grid = player_count, dealer_count, value

    # 为绘图创建策略网格
    policy_grid = np.apply_along_axis(
        lambda obs: policy[(obs[0], obs[1], usable_ace)],
        axis=2,
        arr=np.dstack([player_count, dealer_count]),
    )
    return value_grid, policy_grid


def create_plots(value_grid, policy_grid, title: str):
    """使用值和策略网格创建绘图。"""
    # 创建一个新图形，包含2个子图（左：状态值，右：策略）
    player_count, dealer_count, value = value_grid
    fig = plt.figure(figsize=plt.figaspect(0.4))
    fig.suptitle(title, fontsize=16)

    # 绘制状态值
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1.plot_surface(
        player_count,
        dealer_count,
        value,
        rstride=1,
        cstride=1,
        cmap="viridis",
        edgecolor="none",
    )
    plt.xticks(range(12, 22), range(12, 22))
    plt.yticks(range(1, 11), ["A"] + list(range(2, 11)))
    ax1.set_title(f"状态值: {title}")
    ax1.set_xlabel("玩家点数")
    ax1.set_ylabel("庄家明牌")
    ax1.zaxis.set_rotate_label(False)
    ax1.set_zlabel("价值", fontsize=14, rotation=90)
    ax1.view_init(20, 220)

    # 绘制策略
    fig.add_subplot(1, 2, 2)
    ax2 = sns.heatmap(policy_grid, linewidth=0, annot=True, cmap="Accent_r", cbar=False)
    ax2.set_title(f"策略: {title}")
    ax2.set_xlabel("玩家点数")
    ax2.set_ylabel("庄家明牌")
    ax2.set_xticklabels(range(12, 22))
    ax2.set_yticklabels(["A"] + list(range(2, 11)), fontsize=12)

    # 添加图例
    legend_elements = [
        Patch(facecolor="lightgreen", edgecolor="black", label="要牌"),
        Patch(facecolor="grey", edgecolor="black", label="停牌"),
    ]
    ax2.legend(handles=legend_elements, bbox_to_anchor=(1.3, 1))
    return fig


# 有A情况下的状态价值和策略（A计为11点）
value_grid, policy_grid = create_grids(agent, usable_ace=True)
fig1 = create_plots(value_grid, policy_grid, title="有可用A")
plt.show()

# %%
# .. image:: /_static/img/tutorials/blackjack_with_usable_ace.png
#

# 无A情况下的状态价值和策略（A计为1点）
value_grid, policy_grid = create_grids(agent, usable_ace=False)
fig2 = create_plots(value_grid, policy_grid, title="无可用A")
plt.show()

# %%
# .. image:: /_static/img/tutorials/blackjack_without_usable_ace.png
#
# 在脚本结束时调用env.close()是个好习惯，
# 这样可以关闭环境使用的任何资源。
#

# %%
# 你觉得你能做得更好吗？
# ------------------------------

# 你可以使用play函数可视化环境
# 并尝试赢得几局游戏。


# %%
# 希望本教程能帮助你掌握如何与OpenAI-Gym环境交互，
# 并开启你解决更多强化学习挑战的旅程。
#
# 建议你通过自己动手来解决这个环境（基于项目的学习除了非常有效！）。
# 你可以应用你最喜欢的离散强化学习算法，或者尝试蒙特卡洛ES方法（在[Sutton & Barto]中介绍，第5.3节），
# 这样你可以直接将自己的结果与书中内容进行比较。
#
# 祝你好运！