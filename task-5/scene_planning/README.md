# 家庭环境中的大模型具身决策系统

本项目提供了一个基于大语言模型（LLM）的具身决策框架，用于在家庭环境中实现机器人任务规划与执行。系统结合了Habitat-Sim仿真环境和partnr-planner框架，支持多种任务场景和配置模式。

项目文件结构

```bash
. # 项目根目录，运行脚本时应处于该目录，在本文档中常用 <root> 表示
├── conf  # hydra config files
│   ├── example_simple_planner.yaml         # 示例 planner 配置
│   └── submit.yaml                         # 主配置文件
├── data  # 数据集与资源文件
├── python_package
│   └── embodiment # 项目核心代码
│       ├── runner.py               # 环境初始化与任务执行工具
│       ├── planner.py              # 任务规划核心实现
│       └── dummy_planner.py        # 示例规划器
├── scripts # 独立 python 脚本
│   ├── test_installation.py        # 环境验证脚本
│   └── evaluation.py               # 任务评估脚本
├── start_llm.sh    # LLM 服务部署脚本
└── setup.py        # 项目安装配置
```


## 1. 环境安装与配置

在本文档中，将项目根目录，即 README.md 所在目录称为 `<root>`。

### 1.1 本地部署

环境需求：
- Linux OS
- Python 3.9

#### 1.1.1 环境安装

创建 conda 环境，如果你不使用 conda 环境请跳过这一步。
```bash
conda create -n habitat-llm  python=3.9.2 cmake=3.14.0 -y
conda activate habitat-llm
```

<!-- 安装 CUDA 与 torch，CUDA安装参考 [NVIDIA CUDA Downloads](https://developer.nvidia.com/cuda-downloads)，torch 安装参考 [PyTorch Start Locally](https://pytorch.org/get-started/locally/)。测评所使用环境为 CUDA 12.4, torch 2.5.1。 -->

安装 partnr benchmark

```bash
conda install habitat-sim=0.3.3 withbullet headless -c conda-forge -c aihabitat -y
# 请务必从以下 repo 以及 branch 获取 partnr-planner
git clone https://github.com/RoboticSJTU/partnr-planner.git
cd partnr-planner
git submodule sync
git submodule update --init --recursive
pip install -e ./third_party/habitat-lab/habitat-lab
pip install -e ./third_party/habitat-lab/habitat-baselines
pip install -e ./third_party/transformers-CFG
pip install -r requirements.txt
pip install .
```

安装本项目 package

```bash
cd <root> # 进入项目根目录，即本文档所在目录
pip install --no-build-isolation -e .
```

#### 1.1.2 数据获取

下载数据集和资源文件，通过以下指令解压

```bash
tar -xzvf habitat-partnr.tar.gz
```

将解压后得到的 data 文件夹放到项目根目录下。

#### 1.1.3 环境测试

在项目根目录执行脚本

```bash
python scripts/test_installation.py
```

如果环境安装成功，你会在 `ouputs/habitat_llm/<timestamp>-ci.json/` 下看到运行的结果，在 `videos` 文件夹下能看到渲染出的视频。需要注意的是该视频所演示的场景与题目不同，仅作为环境测试使用。

## 2. LLM 部署

这里使用 [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) 演示模型部署。

### 2.1 权重下载

通过以下指令下载模型权重，你可以在本地执行。

```bash
pip install -U "huggingface_hub[cli]"
# 你可以通过环境变量 HF_HOME 来设置下载目录，默认为 ~/.cache/huggingface
huggingface-cli download Qwen/Qwen3-8B
```

下载完成后可以在本地 `~/.cache/huggingface` 看到下载的文件。将 huggingface 缓存上传到交互式建模环境中的缓存目录 `~/.cache/huggingface`。

### 2.2 vllm 部署

修改 [start_llm.sh](./start_llm.sh) 文件为你使用的模型名称和部署参数，然后运行

```bash
bash start_llm.sh
```

测试大模型部署，注意将模型名称替换成你实际部署的模型。

```bash
# 在新的命令行测试部署情况
curl -X POST "http://localhost:8000/v1/chat/completions" \
	-H "Content-Type: application/json" \
	--data '{
		"model": "Qwen/Qwen3-8B",
		"messages": [
			{
				"role": "user",
				"content": "What is the capital of France?"
			}
		]
	}'
```

如果需要多 GPU 部署，可以在 vllm serve 指令中通过 `--tensor-parallel-size` 指明 GPU 数量。

检查 [`env.bash`](env.bash) 文件，确保其中 `OPENAI_ENDPOINT` 所使用的端口与你部署的相同，且 `OPENAI_MODEL_NAME` 为你部署的模型。

每次新建命令行时，如果要测试 LLM Planner，都需要以下指令来设置环境变量。

```bash
# at <root>
source env.bash
```

## 3. 文档

你可以通过两部分文档来掌握完成本题目所需要使用的各项工具。

**Tutorial Notebooks**

项目根目录下提供了数个 `.ipynb` 文件，可以交互式的运行示例代码，帮助你快速掌握相关 package 的使用，和 Planner 核心接口定义。

**API Documentation**

大多数本项目会用到的 partnr/habitat 接口文档，以及项目提供的脚本接口文档，和 Hydra Configuration System 的基本说明都包含在其中。你可以将其作为 Tutorial Notebooks 的补充资料，以及解题过程中的速查文档。

```bash
pip install mkdocs
mkdocs serve
# Open serving url in your browser.
# It would be http://127.0.0.1:8000/ by default.
```

## 4. 测评

在 `data/datasets/validation_episodes.json.gz` 提供了 validation dataset，你可以通过以下指令，使用该 dataset 对你的 Planner 进行测评。

```bash
python scripts/evaluation.py
```

在 `data/datasets/partnr_episodes/v0_0` 路径下是 partnr benchmark 所提供的所有 episode 数据，如果你想使用其中的特定数据集进行测评，可以通过以下指令

```bash
python scripts/evaluation.py habitat.dataset.data_path=data/datasets/partnr_episodes/v0_0/{数据集名称}.json.gz
```

测评代码会使用 [conf/submit.yaml](./conf/submit.yaml) 文件中的配置运行，请将你所使用的 Planner 以及其参数正确配置在 submit.yaml 文件中。

## 5. 快速开始指南

1. 安装项目依赖：
   ```bash
   pip install --no-build-isolation -e .
   ```

2. 部署 LLM 服务：
   ```bash
   bash start_llm.sh
   source env.bash
   ```

3. 运行示例任务：
   ```bash
   python scripts/evaluation.py
   ```
