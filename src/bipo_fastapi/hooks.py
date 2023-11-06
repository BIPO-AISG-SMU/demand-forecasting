from kedro.framework.hooks import hook_impl
import kedro.framework.hooks.manager as hook
from kedro.pipeline.node import Node
from kedro.framework.project import settings
from kedro.config import ConfigLoader
import os
from typing import Any, Dict
from typing import Union
from bipo_fastapi.load_data_catalog import load_data_catalog
import logging
from datetime import datetime

conf_loader = ConfigLoader(conf_source=settings.CONF_SOURCE)
conf_inference = conf_loader.get("inference*")


class ParamsHooks:
    """Hooks related to parameters."""

    @hook_impl
    def before_node_run(self, node: Node, inputs: Dict[str, Any]) -> Union[dict, None]:
        """Overwrite the start_date and end_date parameters before running the node with modify_params tag. Instantiate logging

        Args:
            node (Node): Pipeline node
            inputs (Dict[str, Any]): Node inputs

        Returns:
            Union[dict, None]: Overwrites node inputs by returning a dictionary if the node tags is "modify_params". Otherwise, returns None
        """
        catalog = load_data_catalog()
        if "modify_params" in node.tags:
            # use the first date and last date of the inference period as values to overwrite the start_date and end_date parameters for the function create_mkt_campaign_counts_start_end
            outlet_df = catalog.load("outlet_df")
            inputs["parameters"]["start_date"] = outlet_df.index[0]
            inputs["parameters"]["end_date"] = outlet_df.index[-1]
            # overwrite the node's input which is called parameters
            return {"parameters": inputs["parameters"]}
        # logging
        load_kedro_logger()
        return None


def load_kedro_logger():
    """Load kedro logging configuration dictionary.

    Returns:
        None
    """
    conf_loader = ConfigLoader(conf_source=settings.CONF_SOURCE)
    config_dict = conf_loader.get("logging*")
    file_handler_list = [
        "info_file_handler",
        "debug_file_handler",
        "error_file_handler",
    ]

    for file_handler_type in file_handler_list:
        # Assumes of the format logs/info.log or similar structure
        level_info_name = config_dict["handlers"][file_handler_type]["level"]
        # Get current run date in ddmmyyyy format
        today = datetime.today()
        today_str = today.strftime("%d/%m/%Y").replace("/", "")
        # Construct the required format

        # Make parent folder containing submodule name if avail, regardless of existence as long submodule_name.is not None
        log_dir_for_submodule = "logs"
        log_filename = f"{today_str}_{level_info_name.lower()}.log"

        orig_log_file = config_dict["handlers"][file_handler_type]["filename"]
        # Delete default configure log filepaths to prevent unnecessary confusion between generated logs
        if os.path.isfile(orig_log_file):
            os.remove(orig_log_file)
        # Construct relevant logfile name based on kedro logging structure and update config path
        rel_logfile_path = os.path.join(log_dir_for_submodule, log_filename)
        config_dict["handlers"][file_handler_type]["filename"] = rel_logfile_path
    # Apply the configuration change assuming it has config
    try:
        logging.config.dictConfig(config_dict)
    except KeyError:
        pass
    return None


# Create hook_manager which is used as input in the kedro runner object
hook_manager = hook._create_hook_manager()
hook_manager.register(ParamsHooks())

# Load kedro config which will be called in init.py
load_kedro_logger()
