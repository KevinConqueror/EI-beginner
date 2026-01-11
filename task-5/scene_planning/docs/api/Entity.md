# class Entity

[Source Code](https://github.com/RoboticSJTU/partnr-planner/blob/habitat_llm/world_model/entity.py)

The node class of [WorldGraph](WorldGraph.md).

It has many subclasses representing different node types. The WorldGraph may contain the following Entity types:

```python
from habitat_llm.world_model.entity import (
    House,
    Room,
    Receptacle,
    Object,
    SpotRobot,
    Human,
)

from habitat_llm.world_model.entities.furniture import Furniture
from habitat_llm.world_model.entities.floor import Floor  # inherited from Furniture
```

This project does not involve human-robot collaboration, so human nodes have no influence.

## Members

### name: str
### properties: Dict[str, Any]

1. `{is_articulated: True}`  
   This property of a `Furniture` Entity indicates whether the furniture can be opened.

2. `{components: ["faucet"]}`  
   An Entity with `"faucet"` components allows you to perform fill or pour actions near it.

3. `{states: {"is_clean": False}}`  
   An Entity with `"is_clean"` states allows you to apply a clean action to it.