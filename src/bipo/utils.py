from pathlib import Path
import logging
import os
from datetime import date, datetime
import pandas as pd
from kedro.io import DataCatalog


# Third Party Imports
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

# Find the Kedro project root directory
def get_project_path():
    """Function that extracts the project root directory in absolute path. Look for "pyproject.toml" to return the project root directory

    Raises:
        RuntimeError: _description_

    Returns:
        project root directory
    """
    current_path = Path.cwd()
    while current_path != Path("/"):
        if (current_path / "src").is_dir():
            logging.info(f"Identified project path: {current_path}")
            return current_path
        current_path = current_path.parent  # Change to parent path
    raise RuntimeError("Kedro project root not found.")