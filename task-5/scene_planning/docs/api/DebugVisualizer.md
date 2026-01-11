# class DebugVisualizer

[Source Code](https://github.com/facebookresearch/habitat-lab/blob/main/habitat-lab/habitat/sims/habitat_simulator/debug_visualizer.py)

```python
from habitat.sims.habitat_simulator.debug_visualizer import DebugVisualizer
```

Support class for simple visual debugging of a Simulator instance.

Use this class for debugging only. Do not use it in your submitted Planner implementation.

## Methods

### __init__

```python
def __init__(
        self,
        sim: habitat_sim.Simulator,
        output_path: str = "visual_debug_output/",
        resolution: Tuple[int, int] = (500, 500),
        clear_color: Optional[mn.Color4] = None,
        equirect=False,
    ) -> None:
```

parameters:

- sim: Simulator instance must be provided for attachment. You can get it from [EnvironmentInterface](https://github.com/RoboticSJTU/partnr-planner/blob/habitat_llm/agent/env/environment_interface.py).sim.
- output_path: Directory path for saving debug images and videos.
- resolution: The desired sensor resolution for any new debug agent (height, width).
- equirect: Optionally use an Equirectangular (360 cube-map) sensor.

You can initialize a DebugVisualizer through

```python
# runner is an instance of PartnrRunner
visualizer = DebugVisualizer(runner.env_interface.sim)
```

### peek

```python
def peek(
    self,
    subject=Union[
        habitat_sim.physics.ManagedArticulatedObject,
        habitat_sim.physics.ManagedRigidObject,
        str,
        int,
    ],
    cam_local_pos: Optional[mn.Vector3] = None,
    peek_all_axis: bool = False,
    debug_lines: Optional[List[Tuple[List[mn.Vector3], mn.Color4]]] = None,
    debug_circles: Optional[
        List[Tuple[mn.Vector3, float, mn.Vector3, mn.Color4]]
    ] = None,
) -> DebugObservation:
        """
        Generic "peek" function generating a DebugObservation image or a set of images centered on a subject and taking as input all reasonable ways to define a subject to peek. Use this function to quickly "peek" at an object or the top-down view of the full scene.

        :param subject: The subject to visualize. One of: ManagedRigidObject, ManagedArticulatedObject, an object_id integer, a string "stage", "scene", or handle of an object instance.
        :param cam_local_pos: Optionally provide a camera location in location local coordinates. Otherwise offset along local -Z axis from the object.
        :param peek_all_axis: Optionally create a merged 3x2 matrix of images looking at the object from all angles.
        :param debug_lines: Optionally provide a list of debug line render tuples, each with a list of points and a color. These will be displayed in all peek images.
        :param debug_circles: Optionally provide a list of debug line render circle Tuples, each with (center, radius, normal, color). These will be displayed in all peek images.
        :return: the DebugObservation containing either 1 image or 6 joined images depending on value of peek_all_axis.
        """
```

This is a very convenient function to look at any object, furniture, etc. in simulation and render an image for it. For example, to get the image of the whole scene

```python
visualizer.peek("scene").get_image()
```

`DebugObservation.get_image()` would return a `PIL.Image.Image` instance which can be directly viewed in jupyter cell.

If you want to peek a world graph entity, you can peek the corresponding sim handle instead.

```python
from habitat_llm.world_model.entities.furniture import Furniture
import magnum as mn
# world_graph is a WorldGraph instance
any_table_entity: Furniture = world_graph.get_node_with_property('type', 'table')
visualizer.peek(any_table_entity.sim_handle, cam_local_pos=mn.Vector3(0,1,0)).get_image()
```

`cam_local_pos` set the local camera position. Note that the distance from camera to target is depended on the size of target bounding box. But you can change the distance by move camera through `translate` or `rotate` method.

Habitat uses a right-handed coordinate system with the Y-axis pointing upward. So `cam_local_pos=mn.Vector3(0,1,0))` means peek from top.

You can peek an object entity in the same way.

But agent node is special, it has a dummy sim_handle in world graph. So you actually can not peek the agent from world graph. If you want to peek the agent, you need to get the sim_handle from simulation directly.

```python
visualizer.peek(runner.env_interface.agents[0].articulated_agent.sim_obj.handle).get_image()
```

### translate

```python
def translate(
    self, vec: mn.Vector3, local: bool = False, show: bool = True
) -> Optional[DebugObservation]:
    """
    Translate the debug sensor agent by a delta vector.

    :param vec: The delta vector to translate by.
    :param local: If True, the delta vector is applied in local space.
    :param show: If True, show the image from the resulting state.
    :return: if show is selected, the resulting observation is returned. Otherwise None.
    """
```

Make sure to set `show=False` if you use this function on a headless cloud server.

The following example code move closer to the scene after peek

```python
visualizer.peek("scene")
visualizer.translate(mn.Vector3(0, -5, 0),show=False)
visualizer.get_observation().get_image()
```

### rotate

```python
def rotate(
    self,
    angle: float,
    axis: Optional[mn.Vector3] = None,
    local: bool = False,
    show: bool = True,
) -> Optional[DebugObservation]:
    """
    Rotate the debug sensor agent by 'angle' radians about 'axis'.

    :param angle: The angle of rotation in radians.
    :param axis: The rotation axis. Default Y axis.
    :param local: If True, the delta vector is applied in local space.
    :param show: If True, show the image from the resulting state.
    :return: if show is selected, the resulting observation is returned. Otherwise None.
    """
```

## Debug Draw Functions

There are also helper functions to draw debug geometries. Note that any drawn geometries will be cleared after the next rendering.

### dblr_draw_bb

```python
def dblr_draw_bb(
    debug_line_render: habitat_sim.gfx.DebugLineRender,
    bb: mn.Range3D,
    transform: mn.Matrix4 = None,
    color: mn.Color4 = None,
) -> None:
    """
    Draw an optionally transformed bounding box with the DebugLineRender interface.

    :param debug_line_render: The DebugLineRender instance.
    :param bb: The local bounding box to draw.
    :param transform: The local to global transformation to apply to the local bb.
    :param color: Optional color for the lines. Default is magenta.
    """
    ...
```

Example code for drawing the bounding box of an object in scene and its related furniture

```python
import habitat.sims.habitat_simulator.sim_utilities as sutils
from habitat.sims.habitat_simulator.debug_visualizer import dblr_draw_bb

any_obj_entity = world_graph.get_all_objects()[0]
related_furniture = world_graph.find_furniture_for_object(any_obj_entity)

obj_in_sim = sutils.get_obj_from_handle(runner.env_interface.sim, any_obj_entity.sim_handle)
furniture_in_sim = sutils.get_obj_from_handle(runner.env_interface.sim, related_furniture.sim_handle)

# peek before debug draw
visualizer.peek(related_furniture.sim_handle, cam_local_pos=mn.Vector3(0,1,0))
dblr_draw_bb(runner.env_interface.sim.get_debug_line_render(),
             obj_in_sim.aabb,
             obj_in_sim.transformation,
             color=mn.Color4(1, 0, 0, 1),)
dblr_draw_bb(runner.env_interface.sim.get_debug_line_render(),
                furniture_in_sim.aabb,
                furniture_in_sim.transformation,
                color=mn.Color4(0, 1, 0, 1),)
# lift camera
visualizer.translate(mn.Vector3(0, 1, 0),show=False)
visualizer.get_observation().get_image()
```