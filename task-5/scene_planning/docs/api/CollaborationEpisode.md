# class [CollaborationEpisode](https://github.com/RoboticSJTU/partnr-planner/blob/habitat_llm/agent/env/dataset.py)

```python
from habitat_llm.agent.env.dataset import CollaborationEpisode
```

Defines a rearrangement task, which includes the instruction, evaluation data, and initial states.

Its members are shown here to help you debug your planner, for example, to identify why some episodes fail. <span style="color:red">**However, your planner implementation must only use the instruction from the episode and no other information.**</span>

Besides, this project only uses one agent for planning. So there is no "Collaboration".

## Members

### instruction: str

Language instruction for current task.

### episode_id: str

Note that it is a string instead of an integer.

### evaluation_propositions: List[EvaluationProposition]

Evaluation propositions. These contain the goal states.

### evaluation_proposition_dependencies: List[EvaluationPropositionDependency]

The dependencies between evaluation propositions. A dependency establishes that a proposition will not be considered for satisfaction unless a "depends_on" proposition has some particular satisfaction state.

### evaluation_constraints: List[EvaluationConstraint]

A constraint is applied over propositions. Examples include temporal constraints and tied quantification.

## Example

Refer to [Partnr Doc: Habitat-LLM Evaluation Engine](https://github.com/RoboticSJTU/partnr-planner/blob/habitat_llm/agent/env/evaluation/README.md) for more information about evaluation functions.

Take episode with id `"919"` for example

instruction: `Clean the shelves in the bedroom. Move the book and clock to the bedroom from the living room. Place the clock on the shelves in the bedroom and turn it on. Place the book next to the clock`

evaluation_propositions:

- '0': one of shelves in scene is clean.
- '1': `book_0` is in bedroom. Note that the entity name is not part of the episode info. The simulation environment is responsible for grounding the object handle to the world graph entity.
- '2': `clock_1` is in bedroom.
- '3': `clock_1` is on top of one shelve.
- '4': `clock_1` is powered on.
- '5': `book_0` is next to `clock_1` within distance 1.0 meters.

evaluation_proposition_dependencies:

- `proposition_indices=[5], depends_on=[1, 2, 3], relation_type='while_satisfied', dependency_mode='any'`.

evaluation_constraints:

- TemporalConstraint. Empty.
- TerminalSatisfactionConstraint.`{'proposition_indices': [0, 1, 2, 3, 4, 5], 'n_propositions': None}`. That means all 5 proposititons should be satisfied at the end of episode.
