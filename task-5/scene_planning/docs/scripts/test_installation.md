# test_installation.py

**Description:**
Validate the installation of the whole environment.

**Usage:**

```bash
python scripts/test_installation.py
```

By default, it uses the Hydra configuration file `conf/baselines/heuristic_full_obs` for testing. Note that this configuration file is intended solely for installation testing, as the environment and configuration differ significantly from those used for evaluation.

To test with a different configuration file, such as one using the example LLM planner, you can specify it with the `--config-name` option:

```bash
python scripts/test_installation.py --config-name example_simple_planner
```

The output will be stored in directory `outputs/validation_episodes.json.gz/`.