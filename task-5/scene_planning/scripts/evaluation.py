# 请勿修改此文件，测评时将会替换为原文件。

import logging
import hydra
from embodiment.runner import PartnerRunner
from habitat_llm.utils import cprint

@hydra.main(
    config_path="../conf",
    config_name="submit"
)
def main(config):  # Hydra injects config here
    # config["habitat"]["dataset"]["data_path"] = ""
    # config["evaluation"]["save_video"] = True
    # agents_order = sorted(
    #     config["habitat"]["simulator"]["agents"].keys()
    # )
    # config["habitat"]["simulator"]["agents_order"] = agents_order
    assert config.world_model.partial_obs == True, "Please set partial_obs to True in the config file."
    assert config.evaluation.type == "decentralized", "Please set evaluation type to decentralized in the config file."
    planner = PartnerRunner(config)  # Explicitly pass config to class
    planner.run_eval()


if __name__ == "__main__":

    logging.getLogger("httpx").setLevel(logging.ERROR)
    cprint(
        "\nStart of the example program to demonstrate long-horizon task planning demo.",
        "green",
    )
    # Run planner
    main()