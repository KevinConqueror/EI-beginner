# OpenVLA 推理示例

本项目演示了如何使用 OpenVLA 模型进行机器人动作预测。OpenVLA 是一个视觉语言动作模型，可以根据图像和自然语言指令预测机器人应采取的动作。

## 功能特点

- 加载 OpenVLA 7B 模型进行视觉语言动作预测
- 支持从文件加载图像进行推理
- 模拟机器人控制接口
- 提供 Python 脚本和 Jupyter Notebook 两种使用方式

## 安装依赖

```bash
pip install flash-attn
tim==0.9.16
torch
transformers
Pillow
numpy
```

## 使用方法

### 方法一：运行 Python 脚本

```bash
python openvla_inference.py
```

### 方法二：使用 Jupyter Notebook

```bash
jupyter notebook openvla_inference.ipynb
```

## 配置要求

- GPU (推荐 A100 或同等性能显卡)
- CUDA 12.4+
- 至少 40GB 显存

## 代码结构

- `openvla_inference.py`: 主要的推理脚本，包含模型加载、图像处理和动作预测功能
- `openvla_inference.ipynb`: Jupyter Notebook 版本，包含详细的执行步骤和输出结果

## 示例指令

```python
instruction = "pick up the red block and place it on the blue square"
```

预测的动作输出格式为 7 维向量：
- 前 3 个元素：Δx, Δy, Δz (位置变化)
- 中间 3 个元素：Δroll, Δpitch, Δyaw (姿态变化)
- 最后 1 个元素：gripper (夹爪状态)

## 注意事项

- 需要 Hugging Face 认证令牌来下载模型权重
- 确保有足够显存来加载 7B 参数的模型
- 图像文件 `scene.jpg` 需要与脚本在同一目录下