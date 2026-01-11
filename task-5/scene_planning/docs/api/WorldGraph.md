# class WorldGraph

[Source Code](https://github.com/RoboticSJTU/partnr-planner/blob/habitat_llm/world_model/world_graph.py)

```python
from habitat_llm.world_model.world_graph import WorldGraph
```

A semantic representation of the simulation environment. It contains the information about

- Robot state.
- Object state.
- Relation ship between entities (robot and object, object and furniture, etc.)

## Members

### graph: Dict[Entity, Dict[Entity, str]]

Inner dict data structure.

WorldGraph uses a Python dict to store all the edges. Use the following code to traverse them:

```python
# world_graph is a WorldGraph instance
num_edges = sum(len(children_dict) for children_dict in world_graph.graph.values())  
print("Number edges: ",num_edges)  
print_format = "{:<30}{:<15}{:<30}"
for parent_node, children_dict in world_graph.graph.items():
    for child_node, edge in children_dict.items():
        print(print_format.format(parent_node.name, edge, child_node.name))
```

Each node is an instance of [Entity](./Entity.md). Most semantic information is contained within the `Entity` class.

## Methods

### to_string

```python
def to_string(compact: bool = False) -> str:
    ...
```

Convert WorldGraph to a string with the format:

```
Room: bathroom_1
    SpotRobot: agent_0
    Furniture: table_37
        Receptacle: rec_table_37_0
            Object: kettle_3
```

Parameters:

- `compact: bool`: Whether to include Receptacle or not. Default is `False`.

### get_neighbors

```python
def get_neighbors(self, node: Union[Entity, str]) -> Dict[Entity, str]:
```

Parameters:

- `node: Union[Entity, str]`: Current node. If it is a string, it will first fetch the corresponding node with `get_node_from_name`.

Returns:

- `Dict[Entity, str]`: {neighbor, relation}

### get_neighbors_of_type

```python
def get_neighbors_of_type(self, node: Union[Entity, str], class_type) -> List[Entity]:
```

Parameters:

- `node: Union[Entity, str]`: Same as above.
- `class_type`: [Entity](./Entity.md) or any subclass of it.

Returns:

- `List[Entity]`: Note that the return value is different from `get_neighbors`.

### get_all_objects

Get all entities of the Object type.

Similarly, the following interfaces are available:

```python
def get_all_rooms(self):
def get_all_receptacles(self):
def get_all_furnitures(self):
def get_all_objects(self):
def get_spot_robot(self):
```

### get_node_from_name

```python
def get_node_from_name(self, node_name: str) -> Entity:
```

Search for the node with the name `node_name`.

### get_node_with_property

```python
def get_node_with_property(self, property_key, property_val) -> Entity:
```

Search for the **first** node with a specific property value.

### get_room_for_entity

```python
def get_room_for_entity(self, entity) -> Entity:
```

Get the room that contains the specified entity.

### get_closest_object_or_furniture

```python
def get_closest_object_or_furniture(
        self, obj_node, n: int, dist_threshold: float = 1.5
    ) -> List[Union[Object, Furniture]]:
```

Get up to `n` Object or Furniture entities within the distance `dist_threshold` to `obj_node`.



### get_closest_entities

An advanced version of `get_closest_object_or_furniture`.

```python
def get_closest_entities(
    self,
    n: int,
    object_node: Entity = None,
    location: list = None,
    dist_threshold: float = 1.5,
    include_rooms: bool = False,
    include_furniture: bool = True,
    include_objects: bool = True,
) -> List[Union[Object, Furniture, Room]]:
    """
    This method returns n closest objects or furnitures to the given object node, or
    given location, within a distance threshold from the given input.
    If dist_threshold is negative or zero, it returns n closest entities regardless
    of distance.
    """
```

### is_object_with_robot

```python
def is_object_with_robot(self, obj: Union[Entity, str]) -> bool:
```

This method checks if the object is grasped by robot.

### find_furniture_for_object

This method returns Furniture associated with the given object

```python
def find_furniture_for_object(self, obj: Object, verbose: bool = False) -> Optional[Furniture]:
```

Similarly, the following interfaces are available:

```python
def find_receptacle_for_object(self, obj):
def find_furniture_for_receptacle(self, rec):
```

### get_subgraph

```python
def get_subgraph(self, nodes_in, verbose: bool = False):
    """
    Method to get subgraph over objects in the view and agents.
    The relevant subgraph is considered the path from object to closest furniture,
    from agent to object-in-hand and from agent to the room they are in.

    Input is a list of name of entities in the agent's view. We sort through them and
    only keep objects. We then find a path from each object to the first Furniture node,
    which is called that object's relevant-subgraph. This relevant subgraph is then
    used to add/update objects in the world graph.
    """
```