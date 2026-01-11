from habitat_llm.planner import Planner
from typing import Dict, Any, Tuple
from habitat_llm.world_model.world_graph import WorldGraph
import queue
from IPython.display import display, clear_output
from PIL import Image, ImageDraw, ImageFont
from typing import Union, Callable, Optional
from numpy.typing import NDArray
import numpy as np
import cv2

def display_image_in_cell(image: Union[Image.Image, NDArray[np.uint8]]):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    clear_output(wait=True)  # Clear the previous output
    display(image)  # Display the current image
    
def add_text_to_img(img: Union[Image.Image, NDArray[np.uint8]], text: str, size_ratio: float = 0.1, position = (10,10), color="white") -> Image:
    """
    在图片上添加文本

    Args:
        img (Image): 图片
        text (str): 文本
        size_ratio (float, optional): 字体大小与图片大小的比例. Defaults to 0.1.

    Returns:
        Image: 添加文本后的图片
    """
    # Create a drawing context
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    elif not isinstance(img, Image.Image):
        raise ValueError("img must be a PIL Image or a numpy array")
    draw = ImageDraw.Draw(img)

    # Optionally, define font (default font will be used if not specified)
    font_size = int(min(img.size) * size_ratio)
    font = ImageFont.load_default(font_size)  # Load default font with specified size

    # Draw text on the image
    draw.text(position, text, fill=color, font=font)  # Use the specified font

    return img

def closure_append_image_to_video(
    output_video: str, 
    fps: float = 30
) -> Callable[[Optional[Union[Image.Image, NDArray[np.uint8]]]], int]:
    
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = None
    frame_count = 0

    def append_image(image: Union[Image.Image, NDArray[np.uint8], None]) -> int:
        nonlocal writer, frame_count
        if image is None:
            if writer is not None:
                writer.release()
                writer = None
            return frame_count
        if writer is None:
            if isinstance(image, Image.Image):
                width, height = image.size
            else:
                height, width = image.shape[:2]
            writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
        if isinstance(image, Image.Image):
            frame_bgr = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        else:
            frame_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        writer.write(frame_bgr)
        frame_count += 1
        return frame_count

    return append_image

def object_names_str_from_world_graph(world_graph: WorldGraph) -> str:
    object_names = []
    for obj_entity in world_graph.get_all_objects():
        if obj_entity.name not in object_names:
            object_names.append(obj_entity.name)
    return ',\n'.join(object_names)

def high_level_action_to_str(high_level_action: Dict[int, Tuple[str, str, str]]) -> str:
    """
    Convert high level action to string.
    """
    
    action_str = ""
    if len(high_level_action) == 0:
        return "No action"
    for agent_id, action in high_level_action.items():
        if len(action) != 3:
            action_str += f"Invalid action: {action}\n"
        else:
            action_str += f"{agent_id}: {action[0]}[{action[1]}]\n"
    return action_str

class DummyPlanner(Planner):
    """
    A dummy planner that does nothing.
    """
    
    q: queue.Queue
    
    def __init__(self, plan_config, env_interface) -> None:
        super().__init__(plan_config, env_interface)
        self.action = {}
        self.is_done = False
        self.replanning_count = 0
        self.q = None
        self.last_response = {}
        
    def set_image_queue(self, q: queue.Queue):
        """
        Set the image queue.
        """
        self.q = q

    def get_next_action(self,
                        instruction: str,
                        observations: Dict[str, Any],
                        world_graph: Dict[int, WorldGraph]) -> Tuple[Dict[int, Any], Dict[str, Any], bool]:
        if not self.action:
            return {}, {}, True
        
        info = {}
        
        low_level_actions, responses = self.process_high_level_actions(self.action, observations=observations)
        self.last_response = responses
        if any(responses.values()):
            # previous action was done or failed.
            # plan the next action
            print(responses)
            self.is_done = True
            
        text = f'{high_level_action_to_str(self.action)}{object_names_str_from_world_graph(world_graph[0])}'
        # PartnrRunner 会将 info['image_text'] 中的文本添加视频帧
        info['image_text'] = text
        
        if self.q is not None:
            image = add_text_to_img(
                observations['agent_0_third_rgb'].detach().cpu().numpy()[0], 
                text,
                size_ratio=0.05,
            )
            self.q.put(image)
            
        return low_level_actions, info, self.is_done
    
    def reset(self):
        """
        Reset the planner.
        """
        self.is_done = False
        self.last_response = {}
        