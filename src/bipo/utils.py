from pathlib import Path
import logging
import sys
import os
from datetime import date, datetime
from functools import reduce
import yaml
import pandas as pd

# Third Party Imports
from kedro.io import DataCatalog
from kedro_datasets.pandas import CSVDataSet
import kedro

sys.dont_write_bytecode = True


def get_logger():
    global config_dict
    logging.getLogger(__name__)

    project_path = get_project_path()
    rel_logging_yml_path = os.path.join(project_path, "conf", "base", "logging.yml")
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
            log_dir_for_submodule = "logs"
            # os.makedirs(log_dir_for_submodule, exist_ok=True)

            log_filename = f"{today_str}_{level_info_name.lower()}.log"

            # Construct relevant logfile name based on kedro logging structure and update config path
            rel_logfile_path = f"{log_dir_for_submodule}/{log_filename}"
            config_dict["handlers"][file_handler_type]["filename"] = rel_logfile_path

        # Apply the configuration change assuming it has config
        try:
            logging.config.dictConfig(config_dict)
        except:
            logging.info(
                "Unable to apply changes as it has no configuration. SKipping overwrites"
            )

        return None


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


# Old Data Split is using
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


# Find the Kedro project root directory
def get_project_path():
    """Function that extracts the project root directory in absolute path. Look for "pyproject.toml" to return the project root directory

    _extended_summary_

    Raises:
        RuntimeError: _description_

    Returns:
        project root directory
    """
    current_path = Path.cwd()
    while current_path != Path("/"):
        if (current_path / "pyproject.toml").is_file():
            return current_path
        current_path = current_path.parent
    raise RuntimeError("Kedro project root not found.")


def get_input_output_folder(input_file_path: str, output_file_path: str) -> tuple:
    """Loads the parameters configuration, defines the input and output folders,
    and returns them as a tuple.

    Args:
    input_file_path (str): The filepath of the source data folder
    output_file_path (str): The filepath of the destination data folder

    Returns:
    tuple: a tuple containing the input and output folders as Path objects.
    """
    project_path = get_project_path()
    input_folder = project_path / input_file_path
    output_folder = project_path / output_file_path

    return input_folder, output_folder


def add_dataset_to_catalog(
    catalog: kedro.io.DataCatalog, filename: str, filepath: Path
):
    """Adds a csv or json file to the data catalog.

    Args:
        catalog (kedro.io.DataCatalog): Empty datacatalog to store data instances for loading and saving.
        filename (str): The filename to use as the identifier in the data catalog.
        filepath (Path): The location of the data file.
    """
    dataset = CSVDataSet(filepath=str(filepath))
    if filename not in catalog.list():
        catalog.add(filename, dataset)
        logging.info(f"Added dataset: {filename} to the data catalog.")

    return None


def save_data(
    catalog: kedro.io.DataCatalog, outlet_df: pd.DataFrame, output_filename: str
):
    """Function to save the merged transaction dataset as a csv file in the specified location

    Args:
        outlet_df (pandas.core.frame.DataFrame): Individual outlet dataframe
        output_filename (str): output file path

    Returns:
        None
    """
    if outlet_df is None:
        return None
    else:
        catalog.save(output_filename, outlet_df)
