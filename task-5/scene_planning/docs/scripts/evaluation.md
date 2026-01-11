# evaluation.py

Evaluate the planner. By default, it uses the dataset `data/datasets/validation_episodes.json.gz`.

If you want to evaluate another dataset, you can either:

#### 1. Use another configuration file:

`evaluation.py` uses `conf/submit.yaml` by default. You can create a new configuration file, such as `conf/new_config_name.yaml`, and edit the `data_path` line. Then, run `evaluation.py` with the following command:

```bash
python scripts/evaluation.py --config-name new_config_name
```

#### 2. Pass an override parameter directly:

```bash
python scripts/evaluation.py habitat.dataset.data_path=data/datasets/partnr_episodes/v0_0/val.json.gz
```