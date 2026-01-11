# Hydra Configuration System

Refer to [Hydra Doc](https://hydra.cc/docs/intro/) for more details about hydra configuration system. 

Here only the basic usage of Hydra related to this project is introduced.

## Use in Script File

Take `scripts/test_installation.py` as example, there is

```python
@hydra.main(config_path="../conf", config_name="baselines/heuristic_full_obs")
def main(config: DictConfig) -> None:
    ...
```

The `@hydra.main` decorator here specifies

1. The search path of hydra configs (relative to script file path). Here the search path is `../conf` which is the `conf` directory under root directory.
2. The name of config file under search path. The `.yaml` extension can be omitted. Here `conf/baselines/heuristic_full_obs.yaml` would be used.

You can use command line argument `--config-name` to change `config_name` during runtime. For example, to use the example react planner for `test_installation.py`

```bash
python scripts/test_installation.py --config-name example01_react_planner.yaml
```

## Override

Refer [Hydra Doc: Basic Override syntax](https://hydra.cc/docs/advanced/override_grammar/basic/) for overriding the configuration.


