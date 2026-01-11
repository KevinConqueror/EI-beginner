from habitat_llm.examples.setup_env import setup_env, setup_runner
import hydra
from omegaconf import DictConfig
from habitat_llm.evaluation import EvaluationRunner
from habitat_llm.agent.env import EnvironmentInterface
from habitat_llm.agent.env.dataset import CollaborationEpisode

from embodiment.runner import PartnerRunner


@hydra.main(config_path="../conf", config_name="baselines/heuristic_full_obs")
def main(config: DictConfig) -> None:
    """
    Main function to test the installation of the habitat-llm package.
    """
    # Setup the environment
    config.evaluation.save_video = True
    config.world_model.partial_obs = True
    eval_runner: PartnerRunner = PartnerRunner(config)
    episode_id: str = eval_runner.env_interface.env.current_episode().episode_id
    current_episode: CollaborationEpisode = [e for e in eval_runner.env_interface.env.episodes if e.episode_id == episode_id][0]
    eval_runner.run_instruction(current_episode.instruction, "test_installation")

if __name__ == "__main__":
    main()