# OpenPI 推理项目

## 项目概述
本项目实现了基于OpenPI框架的模型推理功能，包含Jupyter Notebook演示和Python脚本实现两种使用方式。

## 文件说明
- `openpi_inference.ipynb`：交互式推理演示（Jupyter Notebook）
- `openpi_inference.py`：命令行推理脚本

## 使用方法

### 方式一：使用Jupyter Notebook
```bash
jupyter notebook openpi_inference.ipynb
```

### 方式二：使用Python脚本
```bash
python openpi_inference.py
```

## 依赖安装
```bash
pip install torch transformers jupyter
```

## 注意事项
1. 确保已下载预训练模型权重
2. 输入数据格式需符合模型要求
3. GPU环境可加速推理过程