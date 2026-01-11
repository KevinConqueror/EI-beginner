# Code as Policies 桌面级任务规划

## 项目概述
本项目基于LLM和代码策略的桌面级任务规划，包含数据处理脚本、可视化demo和自动化资源下载功能。

## 项目结构
```
├── Code_as_Policies_Interactive_Demo.ipynb  # Jupyter交互式演示笔记本
├── code_as_policies_interactive_demo.py  # 核心策略执行Python模块
└── download.sh                         # 依赖资源自动下载脚本
```

## 环境要求
- Python 3.8+
- Jupyter Notebook
- 基础数据科学库（pandas, numpy, matplotlib）

## 快速开始
1. 安装依赖：
```bash
pip install pandas numpy matplotlib shapely pybullet astunparse pygments opencv-python
```

2. 下载演示数据：
```bash
chmod +x download.sh  # Linux/Mac
./download.sh         # 执行下载
```

3. 启动交互式演示：
```bash
jupyter notebook Code_as_Policies_Interactive_Demo.ipynb
```
4. 启动python脚本：
```bash
python code_as_policies_interactive_demo.py
```

## 功能说明
- `code_as_policies_interactive_demo.py`：提供策略解析和执行的核心API
- `download.sh`：自动获取演示所需数据集和预训练模型
- Jupyter Notebook：包含完整的交互式操作流程和可视化示例

## 注意事项
首次运行建议先执行download.sh获取必要资源