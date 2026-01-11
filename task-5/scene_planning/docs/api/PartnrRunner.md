# class PartnrRunner

Source code located at `python_package/embodiment/runner.py`.

A helper class used for setting up the environment and running the planner.

## Read-Only Properties

### planner: Dict[int, [Planner](Planner.md)]

Retrieves the current planners.

- **key**: `int`. Since there is only one configured agent, this will always be `0`.
- **value**: [Planner](Planner.md). The Planner instance.

### episodes: List[[CollaborationEpisode](CollaborationEpisode.md)]

A list of all episodes.

### current_episode_id: str

The ID of the current episode. `PartnrRunner` does not provide a method to access an episode by a specific ID. However, you can retrieve it using the following code:

```python
eps_id = "919"
eps: CollaborationEpisode = [e for e in runner.episodes if e.episode_id == eps_id][0]
```

### hl_action_descriptions: str

A string description of all the high-level actions the agent has.

## Methods

### run_instruction(instruction: str, output_name: str)

Uses the planner to plan for the given instruction. It utilizes the environment of the current episode.

This function can be used to test your planner. Note that due to potential misalignment between the episode environment and the instruction input, the evaluation result may not be reliable.

Parameters:

- instruction: language instruction.
- output_name: as part of output files' name.

### reset(self, episode_id: str=None)

Reset the environment. It will move to the next episode in the dataset by default. If `episode_id` is provided, it will reset the environment to a specific episode.