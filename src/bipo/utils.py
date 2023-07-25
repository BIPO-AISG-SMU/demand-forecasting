from pathlib import Path
import logging
import sys
import os
from datetime import date, datetime
from functools import reduce
import yaml

sys.dont_write_bytecode = True


def get_logger():
    global config_dict
    logging.getLogger(__name__)

    rel_logging_yml_path = os.path.join("conf", "base", "logging.yml")
    with open(rel_logging_yml_path) as f:
        # Load yaml file
        config_dict = yaml.load(f, Loader=yaml.Loader)

        file_handler_list = [
            "info_file_handler",
            "debug_file_handler",
            "error_file_handler",
        ]

        for file_handler_type in file_handler_list:
            # Assumes of the format logs/info.log or similar structure
            # base, extension = os.path.splitext(filename)
            level_info_name = config_dict["handlers"][file_handler_type]["level"]
            # Get current run date in ddmmyyyy format
            today = datetime.today()
            today_str = today.strftime("%d/%m/%Y").replace("/", "")
            # Construct the required format

            # Make parent folder containing submodule name if avail, regardless of existence as long submodule_name.is not None
            log_dir_for_submodule = f"logs/{today_str}"
            os.makedirs(log_dir_for_submodule, exist_ok=True)

            log_filename = f"{level_info_name.lower()}.log"

            # Construct relevant logfile name based on kedro logging structure and update config path
            rel_logfile_path = f"{log_dir_for_submodule}/{log_filename}"
            config_dict["handlers"][file_handler_type]["filename"] = rel_logfile_path
            print(rel_logfile_path)

        # Apply the configuration change
        logging.config.dictConfig(config_dict)

        return logging


def create_dir_from_project_root(config_file_dir_level_lists: list) -> str:
    """Function constructs a directory path by combining the output of get_project_root function and unpacking a list containing the directory levels specified in config file.

    Example:
        The config_file_dir_level_lists provided is ["dir1", "dir2"],
    the directory created would be <project_root absolute_path>/dir1/dir2.

    Args:
        config_file_dir_level_lists (list): List of directories, representing the subsequent levels of directory in folder construct.

    Raises:
        None

    Returns:
        string representing a constructed directory path
    """

    # Add current working directory to the first entry input list. This is to facilitate the unpacking of list elements when constructing directory path
    config_file_dir_level_lists.insert(0, os.getcwd())

    # Unpack list to construct full path
    constructed_path = os.path.join(*config_file_dir_level_lists)

    return constructed_path


def get_project_root() -> Path:
    """Function that extracts the project root directory in absolute path.

    Args:
        None

    Raises:
        None

    Returns:
        Absolute path of project root directory located 3 levels above.
    """
    return Path(__file__).parent.parent.parent.absolute()
