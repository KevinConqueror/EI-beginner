from habitat_llm.planner.llm_planner import LLMPlanner
from habitat_llm.world_model.world_graph import WorldGraph
from typing import Dict, Any, Tuple
from habitat_llm.llm.openai_chat import OpenAIChat
from habitat_llm.llm.instruct.utils import zero_shot_action_parser


prompt_template = """
You are an agent that solves planning problems. The task assigned to you will be situated in a house and will generally involve navigating to objects, picking and placing them on different receptacles to achieve rearrangement. You strictly follow any format specifications and pay attention to the previous actions taken in order to avoid repeating mistakes. 

Rooms do not need to be explored more than once.
This means if you have explored the living room and have not found the object, then you should explore the kitchen, if a relevant object is still not found, you should explore the hallway etc...

Many calls to the same action in a row are a sign that something has gone wrong and you should try a different action.

Current world graph:
{world_graph}

Possible Actions:
- Clean: Used for cleaning an object. You need to provide the name of the object to clean.
- Close: Used for closing an articulated entity. You must provide the name of the furniture you want to close. Example (Close[chest_of_drawers_1])
- Explore: Search a specific room by visiting various receptacles or furnitures in that room. The input to the skill is the EXACT name of the room to be visited. Use the room names provided in the house description. This tool exhaustivly explores the specified room. Example (Explore[kitchen_1])
- Fill: Used for filling an object. You need to provide the name of the object to fill.
- Navigate: Used for navigating to an entity. You must provide the name of the entity you want to navigate to. Example (Navigate[counter_22])
- Open: Used for opening an articulated entity. You must provide the name of the furniture you want to open. Example (Open[chest_of_drawers_1])
- Pick: Used for picking up an object. You must provide the name of the object to be picked. The agent cannot hold more than one object at a time. Example (Pick[cup_1])
- Place: Used for placing an object on a target location. You need to provide the name of the object to be placed, the name of the furniture where it should be placed, spatial relation ("on" or "within") describing the relation between the object and furniture. The object to be placed must already be held by the agent (i.e. picked previously). In addition to these, you can request to place the object near another object. For that you can optionally provide a spatial constraints ("next_to") and the name of the reference object. To place next to an object, the reference object must already be on the target furniture. API template - Place[<object_to_be_placed>, <spatial_relation>, <furniture to be placed on>, <spatial_constraint>, <reference_object>]. spatial_constraint and reference_object should be set to "None" when necessary.
- Pour: Used for pouring from one container to another. This skill will pour into the specified container from whichever container is currently held by the agent.
- PowerOff: Used for turning off a powered object. You need to provide the name of the object to be turned off.
- PowerOn: Used for turning on a powered object. You need to provide the name of the object to be turned on.
- Wait: Used to make agent stay idle for some time. Example (Wait[])
- Done: Used to indicate that the agent has finished the task. Example (Done[])

What is the next action to make progress towards completing the task:

{instruction}

Return your response in the following format

Thought: <reasoning for why you are taking the next action>
<next action call>

Here is an example:
Thought: Since there are no objects found I should explore a room I have not explored yet
Explore[<room name>]

"""

class EmbodimentPlanner(LLMPlanner):
    
    llm: OpenAIChat
    def __init__(self, plan_config, env_interface) -> None:
        super().__init__(plan_config, env_interface)
        self.action = {}
        self.is_done = False
        self.replanning_count = 0
        self.last_response = {}
        
    def prepare_prompt(self, input_instruction: str, world_graph: Dict[int, WorldGraph]) -> str:
        world_graph_str = world_graph[0].to_string()
        
        return prompt_template.format(
            world_graph=world_graph_str,
            instruction=input_instruction
        )
    
    def get_next_action(self,
        instruction: str,
        observations: Dict[str, Any],
        world_graph: Dict[int, WorldGraph]) -> Tuple[Dict[int, Any], Dict[str, Any], bool]:
        
        if not self.action:
            reply = self.llm.generate(self.prepare_prompt(instruction, world_graph))
            action_line = reply.split("\n")[-1]
            self.action = zero_shot_action_parser(
                self.agents, action_line
            )
            self.is_done = "Done[]" in action_line
            print(self.action)
            print(reply)
            self.replanning_count += 1
            
        low_level_actions, responses = self.process_high_level_actions(self.action, observations=observations)
        self.last_response = responses
        if any(responses.values()):
            # previous action was done or failed.
            # plan the next action
            print(responses)
            self.action = {}
            
        if self.replanning_count >= self.planner_config.replanning_threshold:
            self.is_done = True
        
        return low_level_actions, {"high_level_actions": self.action}, self.is_done
    
    def reset(self):
        """
        Reset the planner.
        """
        super().reset()
        self.action = {}
        self.is_done = False
        self.replanning_count = 0
        self.last_response = {}