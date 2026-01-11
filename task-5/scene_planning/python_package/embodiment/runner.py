#!/usr/bin/env python3
# 请勿修改此文件，测评时将会替换为原文件。

from typing import Dict
import hydra
import cv2
import numpy as np
from PIL import Image
import csv
import sys
import time
import copy
import os
import traceback
import json
import shutil
import threading


from tqdm import tqdm
from omegaconf import OmegaConf, DictConfig


from habitat.datasets.rearrange.samplers.receptacle import find_receptacles
from habitat_llm.utils import cprint, setup_config, fix_config
from habitat_llm.agent.env.evaluation.evaluation_functions import (
    aggregate_measures,
)
from habitat_llm.examples.example_utils import DebugVideoUtil
from habitat_baselines.utils.info_dict import extract_scalars_from_info
from habitat_llm.agent.env.dataset import CollaborationDatasetV0
from habitat_llm.evaluation import (
    CentralizedEvaluationRunner,
    DecentralizedEvaluationRunner,
    EvaluationRunner,
)
from habitat_llm.agent.env import (
    EnvironmentInterface,
    register_actions,
    register_measures,
    register_sensors,
    remove_visual_sensors,
)
from habitat_llm.planner.planner import Planner
from PIL import Image, ImageDraw, ImageFont

from typing import Union
from numpy.typing import NDArray

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



def get_output_file(config, env_interface):
    dataset_file = env_interface.conf.habitat.dataset.data_path.split("/")[-1]
    episode_id = env_interface.env.env.env._env.current_episode.episode_id
    output_file = os.path.join(
        config.paths.results_dir,
        dataset_file,
        "stats",
        f"{episode_id}.json",
    )
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    return output_file


# Function to write data to the CSV file
def write_to_csv(file_name, result_dict):
    # Sort the dictionary by keys
    # Needed to ensure sanity in multi-process operation
    result_dict = dict(sorted(result_dict.items()))
    with open(file_name, mode="a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=result_dict.keys())

        # Check if the file is empty (to write headers)
        file.seek(0, 2)
        file_empty = file.tell() == 0
        if file_empty:
            writer.writeheader()

        writer.writerow(result_dict)


def save_exception_message(config, env_interface):
    output_file = get_output_file(config, env_interface)
    exc_string = traceback.format_exc()
    failure_dict = {"success": False, "info": str(exc_string)}
    with open(output_file, "w+") as f:
        f.write(json.dumps(failure_dict))


def save_success_message(config, env_interface, info):
    output_file = get_output_file(config, env_interface)
    failure_dict = {"success": True, "stats": json.dumps(info)}
    with open(output_file, "w+") as f:
        f.write(json.dumps(failure_dict))


# Write the config file into the results folders
def write_config(config):
    dataset_file = config.habitat.dataset.data_path.split("/")[-1]
    output_file = os.path.join(config.paths.results_dir, dataset_file)
    os.makedirs(output_file, exist_ok=True)
    with open(f"{output_file}/config.yaml", "w+") as f:
        f.write(OmegaConf.to_yaml(config))

    # Copy over the RLM config
    planner_configs = []
    suffixes = []
    if "planner" in config.evaluation:
        # Centralized
        if "plan_config" in config.evaluation.planner is not None:
            planner_configs = [config.evaluation.planner.plan_config]
            suffixes = [""]
    else:
        for agent_name in config.evaluation.agents:
            suffixes.append(f"_{agent_name}")
            planner_configs.append(
                config.evaluation.agents[agent_name].planner.plan_config
            )

    for plan_config, suffix_rlm in zip(planner_configs, suffixes):
        if "llm" in plan_config and "serverdir" in plan_config.llm:
            yaml_rlm_path = plan_config.llm.serverdir
            if len(yaml_rlm_path) > 0:
                yaml_rlm_file = f"{yaml_rlm_path}/config.yaml"
                if os.path.isfile(yaml_rlm_file):
                    shutil.copy(
                        yaml_rlm_file, f"{output_file}/config_rlm{suffix_rlm}.yaml"
                    )

# Method to load agent planner from the config


class PartnerRunner:

    timeout: int
    
    def __init__(self, config):
        self.timeout = 60*10 # 10 minutes
        self.config, self.dataset = self.initialize_eval(config)
        self.prepare_planner(self.config, self.dataset)

    def initialize_eval(self, config):
        # fix_config(config)
        seed = int(time.time())
        # Setup config
        config = setup_config(config, seed)
        dataset = CollaborationDatasetV0(config.habitat.dataset)

        write_config(config)

        return config, dataset

    def prepare_planner(self, config, dataset: CollaborationDatasetV0 = None, conn=None):
        if config == None:
            cprint("Failed to setup config. Exiting", "red")
            return

        # Setup interface with the simulator if the planner depends on it
        if config.env == "habitat":
            self.save_video = False
            # Remove sensors if we are not saving video
            keep_rgb = False
            if "use_rgb" in config.evaluation:
                keep_rgb = config.evaluation.use_rgb
            if not config.evaluation.save_video and not keep_rgb:
                remove_visual_sensors(config)
            else:
                self.save_video = True

            # TODO: Can we move this inside the EnvironmentInterface?
            # We register the dynamic habitat sensors
            register_sensors(config)
            # We register custom actions
            register_actions(config)
            # We register custom measures
            register_measures(config)

            # Initialize the environment interface for the agent
            env_interface = EnvironmentInterface(
                config, dataset=dataset, init_wg=False)

            try:
                env_interface.initialize_perception_and_world_graph()
            except Exception:
                print("Error initializing the environment")
                if config.evaluation.log_data:
                    save_exception_message(config, env_interface)
        else:
            env_interface = None
        # Instantiate the agent planner
        eval_runner: EvaluationRunner = None
        if config.evaluation.type == "centralized":
            eval_runner = CentralizedEvaluationRunner(
                config.evaluation, env_interface)
        elif config.evaluation.type == "decentralized":
            eval_runner = DecentralizedEvaluationRunner(
                config.evaluation, env_interface)
        else:
            cprint(
                "Invalid planner type. Please select between 'centralized' or 'decentralized'. Exiting",
                "red",
            )
            return

        os.makedirs(config.paths.results_dir, exist_ok=True)
        self.eval_runner = eval_runner
        self.env_interface = env_interface

        dataset_file = self.env_interface.conf.habitat.dataset.data_path.split(
            "/")[-1]
        results_dir = self.env_interface.conf.paths.results_dir
        self.output_dir = f"{results_dir}/{dataset_file}/"
        os.makedirs(self.output_dir, exist_ok=True)
        self.dvu = DebugVideoUtil(self.env_interface, self.output_dir)

    def step_low_level(self, low_level_actions, planner_info):
        obs, reward, done, info = self.eval_runner.env_interface.step(
            low_level_actions)
        # Refresh observations
        observations = self.eval_runner.env_interface.parse_observations(obs)
        frames_concat = None
        if self.save_video:
            frames_concat = self.dvu._DebugVideoUtil__get_combined_frames(
                observations)
            frames_concat = np.ascontiguousarray(frames_concat)
            text = ""
            if "high_level_actions" in planner_info.keys():
                for idx, action in planner_info["high_level_actions"].items():
                    if action is not None and len(action) > 1:
                        text += f"{idx}: {action[0]}[{action[1]}]\n"
            if "image_text" in planner_info.keys():
                text += planner_info["image_text"]
            if text != "":
                frames_concat = np.array(add_text_to_img(
                    frames_concat,
                    text,
                    size_ratio=0.05,
                    position=(10, 10),
                    color="white",
                ))
        return observations, frames_concat

    def run_instruction(self, instruction, output_name):
        t_0 = time.time()
        total_step_count = 1

        # Reset planners and the agents owned by the planners
        # This will also reset skills owned by the agents to
        # make eval runner ready for next episode
        self.eval_runner.reset_planners()
        self.dvu.frames = []
        # Initialize metadata
        self.eval_runner.initialize_instruction_metadata(
            instruction, output_name)
        # Initialize sensor observations
        observations = self.eval_runner.env_interface.get_observations()
        info = {
            "task_percent_complete": 0.0,
            "task_state_success": 0.0,
            "total_step_count": total_step_count,
        }

        measure_names = [
            "auto_eval_proposition_tracker",
            "task_constraint_validation",
            "task_percent_complete",
            "task_state_success",
            "task_evaluation_log",
            "task_explanation",
        ]
        measures_to_log = [
            "task_percent_complete",
            "task_state_success",
            "task_explanation",
        ]
        planner_info = {}
        low_level_actions = {}
        should_end = False
        # Plan until required
        while not should_end:
            # Execute low level actions
            if len(low_level_actions) > 0:
                observations, frame = self.step_low_level(low_level_actions, planner_info)
                if self.save_video:
                    self.dvu.frames.append(frame)
            # Get next low level actions
            
            # create timer for timeout
            
            
            # low_level_actions, planner_info, should_end = self.eval_runner.get_low_level_actions(
            #     self.eval_runner.current_instruction,
            #     observations,
            #     self.eval_runner.env_interface.world_graph
            # )
            
            timer = threading.Timer(self.timeout - (time.time() - t_0), lambda: setattr(self, 'timeout_reached', True))
            self.timeout_reached = False
            try:
                timer.start()
                low_level_actions, planner_info, should_end = self.eval_runner.get_low_level_actions(
                    self.eval_runner.current_instruction,
                    observations,
                    self.eval_runner.env_interface.world_graph
                )
            except Exception as e:
                if self.timeout_reached:
                    print("Timeout reached. Interrupting get_low_level_actions.")
                else:
                    print("An error occurred:", e)
                low_level_actions, planner_info, should_end = {}, {}, True
            finally:
                timer.cancel()
            
            # We terminate the episode if this loop gets stuck
            curr_env = self.eval_runner.env_interface.env.env.env._env
            if total_step_count > curr_env._max_episode_steps:
                should_end = True
            if should_end:
                measures = curr_env.task.measurements.measures
                for measure_name in measure_names:
                    measures[measure_name].update_metric(
                        task=curr_env.task, episode=curr_env.current_episode
                    )
                for measure_name in measure_names:
                    if measure_name in info:
                        info[measure_name] = measures[measure_name].get_metric()

            planner_info["stats"] = {
                info_name: info[info_name]
                for info_name in measures_to_log
                if info_name in info
            }

            # Add step count to planner_info
            planner_info["total_step_count"] = total_step_count
            planner_info["sim_step_count"] = total_step_count
            # Add world description to planner_info
            # on every replanning step and at the end of planning
            if (
                "replan_required" in planner_info
                and planner_info["replan_required"]
                and any(planner_info["replan_required"].values())
            ) or should_end:
                planner_info["curr_graph"] = {
                    agent_id: self.env_interface.world_graph[agent_id].get_world_descr(
                        is_human_wg=int(
                            agent_id) == self.env_interface.human_agent_uid
                    )
                    for agent_id in range(len(self.eval_runner.agents))
                }

            # Update agent state and action history
            copy_planner_info = copy.deepcopy(planner_info)
            self.eval_runner.update_agent_state_history(copy_planner_info)
            self.eval_runner.update_agent_action_history(copy_planner_info)

            total_step_count += 1

        t_runtime = time.time() - t_0
        info["runtime"] = t_runtime

        # Merge dictionaries
        info |= planner_info
        # Save the video if neccessary
        if self.save_video:
            self.dvu._make_video(
                play=False, postfix=self.eval_runner.episode_filename)

        return info

    def reset(self, episode_id=None):
        self.env_interface.reset_environment(move_to_next_episode=True, episode_id=episode_id)
        self.eval_runner.reset()

    def run_eval(self):
        num_episodes = len(self.env_interface.env.episodes)
        # run episodes in sequence
        stats_episodes: Dict[str, Dict] = {}
        for _ in tqdm(range(num_episodes), total=num_episodes):
            # Get episode id
            episode_id = self.env_interface.env.env.env._env.current_episode.episode_id

            # Get instruction
            instruction = self.env_interface.env.env.env._env.current_episode.instruction
            print("\n\nEpisode", episode_id)

            try:
                info = self.run_instruction(
                    instruction,
                    output_name=f"episode_{episode_id}"
                )

                info_episode = {
                    "episode_id": episode_id,
                    "instruction": instruction,
                }
                stats_keys = {
                    "task_percent_complete",
                    "task_state_success",
                    "total_step_count",
                    "runtime",
                }

                # add replanning counts to stats_keys as scalars if replanning_count is a dict
                if "replanning_count" in info and isinstance(
                    info["replanning_count"], dict
                ):
                    for agent_id, replan_count in info["replanning_count"].items():
                        stats_keys.add(f"replanning_count_{agent_id}")
                        info[f"replanning_count_{agent_id}"] = replan_count

                stats_episode_i = extract_scalars_from_info(
                    info, ignore_keys=info.keys() - stats_keys
                )
                stats_episodes[episode_id] = stats_episode_i
                cprint("\n---------------------------------", "blue")
                cprint(f"Metrics For Run Episode {episode_id}:", "blue")
                for k, v in stats_episodes[episode_id].items():
                    cprint(f"{k}: {v:.3f}", "blue")
                cprint("\n---------------------------------", "blue")
            except Exception as e:
                # print exception and trace
                traceback.print_exc()
                print("An error occurred while running the episode:", e)
                print(f"Skipping evaluating episode: {episode_id}")
                if self.config.evaluation.log_data:
                    save_exception_message(self.config, self.env_interface)

            # Reset evaluation runner
            self.reset()

            # aggregate metrics across the current run.
        run_metrics = aggregate_measures(stats_episodes)
        cprint("\n---------------------------------", "blue")
        cprint(f"Over All Metrics:", "blue")
        for k, v in run_metrics.items():
            cprint(f"{k}: {v:.3f}", "blue")
        cprint("\n---------------------------------", "blue")

        # Write aggregated results across run
        write_to_csv(self.config.paths.run_result_file_path, run_metrics)
        self.env_interface.env.close()


    @property
    def planner(self) -> Dict[int, Planner]:
        return self.eval_runner.planner
    
    @property
    def episodes(self):
        return self.eval_runner.env_interface.env.episodes
    
    @property
    def current_episode_id(self):
        return self.eval_runner.env_interface.env.current_episode().episode_id
        
    @property
    def hl_action_descriptions(self):
        """
        Get the list of tools available in the environment.
        """
        return self.eval_runner.agents[0].tool_descriptions