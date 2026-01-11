# Hydra Configation File

Refer to [Hydra Doc](https://hydra.cc/docs/intro/) for more details.

Take `conf/example_simple_planner.yaml` as example, there is

```yaml
# @package _global_

defaults:
  - /examples/embodiment
  # 使用配置 /planner/llm_planner.yaml 作为基础配置
  - override /planner@evaluation.agents.agent_0.planner: llm_planner
  # 使用配置 /llm/openai_chat.yaml
  # 注意，该配置文件中部分参数会被 env.bash 文件中的环境变量覆盖
  - override /llm@evaluation.agents.agent_0.planner.plan_config.llm: openai_chat

habitat:
  dataset:
    data_path: data/datasets/validation_episodes.json.gz
    
# 修改 /planner/llm_planner.yaml 中的 Planner 类型为示例 Planner
evaluation:
  agents:
    agent_0:
      planner:        
        _target_: 'embodiment.planner.EmbodimentPlanner'
        plan_config:
            replanning_threshold: 3 # max number of times the LLM is allowed to plan to finish the task
```

## Default List

`defaults` defines the [Default List](https://hydra.cc/docs/advanced/defaults_list/) which is a list of input configs to build the final config.

### /examples/embodiment

This line loads the configuration file locatetd at `conf/examples/embodiment.yaml`. 

By default all configs in `embodiment.yaml` are loaded to field `examples.*`. However, the `# @package _global_` directive on the first line of `embodiment.yaml` indicates that the configurations in this file are loaded into the global field.

Actually `/examples/embodiment` can also be a [config store](https://hydra.cc/docs/tutorials/structured_config/config_store/) named `embodiment` registered at group `example`. Partnr/Habitat uses lots of config store and we do not touch them in this project. You can ignore that case.

### override /planner@evaluation.agents.agent_0.planner: llm_planner

This line overrides the configuration under the `evaluation.agents.agent_0.planner` field with the configuration defined in `conf/planner/llm_planner.yaml`.

`@evaluation.agents.agent_0.planner` defines the target field.


## Others

The parts other than the Default List are the configurations newly added or modified in this configuration file.

Note that when changing some configuration field, change it through

```yaml
habitat:
  dataset:
    data_path: data/datasets/validation_episodes.json.gz
```

Instead of

```yaml
habitat.dataset.data_path: data/datasets/validation_episodes.json.gz
```

The latter would delete all other fields under `habitat` field.

### \_target\_: 'embodiment.planner.EmbodimentPlanner'

`_target_` is a special field of hydra configuration. The value of that field should be an importable class. For example, you can import `EmbodimentPlanner` through

```python
from embodiment.planner import EmbodimentPlanner
```

In this example, hydra would instantiate `embodiment.planner.EmbodimentPlanner` object with parameter `plan_config`. 

Note that there is also 

```yaml
_recursive_: False
_partial_: True
```

configured in `conf/planner/llm_planner.yaml` which will influence the instantiating behavoir. See [Hydra Doc: Instantiating objects with Hydra](https://hydra.cc/docs/advanced/instantiate_objects/overview/) for details.
