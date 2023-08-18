import pandas as pd
import os
import logging
import numpy as np
import warnings
import chardet
import openpyxl
import sys
from kedro.config import ConfigLoader
from kedro.framework.project import settings

# import sys
# parent_directory = os.path.dirname(os.path.abspath(__file__)) # current path
# parent_parent_directory = os.path.dirname(parent_directory) # pipelines
# parent_parent_parent_directory = os.path.dirname(parent_parent_directory) # bipo
# sys.path.append(parent_parent_parent_directory)

from bipo.utils import get_project_path

# initiate to constants.yml and parameters.yml
project_path = get_project_path()
conf_loader = ConfigLoader(conf_source=project_path / "conf")
conf_params = conf_loader["parameters"]["dataloader"]
constants = conf_loader.get("constants*")["dataloader"]

logging = logging.getLogger(__name__)

SOURCE_DATA_DIR = constants["data_source_dir"]
ENCODINGTYPE = constants["encodingtype"]  # from validation/dataloader_validation.yaml
EXPECTED_COLUMNS = constants["expected_columns"]
# EXPECTED_SHEETS = params["expected_sheetnames"]
TARGET_FEATURE_NAME = conf_loader.get("constants*")["general"]["target_feature"]


class DataCheck:
    """Performs various checks on the data, which will be inherited by each dataset class.
    Methods:
        1. check_file_format_encoding: Check file's format and encoding type.
        2. check_columns: Check data columns matches the expected columns.
        3. check_data_exists: Check if dataset is empty.()
        4. check_target_features: check if target feature is present. (dataloader.py)
    """

    def __init__(self, outlet: str = None):
        """Initialize with outlet name.

        Args:
            outlet (str): outlet name
            logging: (logging.Logger): logger object defined in utils.py
        """
        self.outlet = outlet

    def check_columns(self, original_columns: list, dataset_name: str) -> None:
        """Check that columns from original dataframe matches the expected columns

        Args:
            original_columns (list): list of original dataset columns
            dataset_name (str): file name in data config file

        Returns:
            None
        """
        original_columns = set(original_columns)
        expected_columns = set(EXPECTED_COLUMNS[dataset_name])

        # Check if got any missing columns
        if expected_columns.difference(
            original_columns
        ):  # if difference is not empty, missing values from original_columns.
            logging.error(
                f"{expected_columns.difference(original_columns)} are missing from original columns",
                stack_info=True,
                exc_info=True,
            )

        # Check if got any additional columns
        elif original_columns.difference(
            expected_columns
        ):  # if difference is not empty, missing values from EXPECTED_COLUMNS.
            logging.error(
                f"{original_columns.difference(expected_columns)} not found in expected columns"
            )
        return None

    def check_data_exists(self, dataframe) -> None:
        """Function that checks if the dataset (raw datasets and merged dataset)is empty. Raises a ValueError to terminate the program if dataset is empty as this is a critical error.

        Args:
            dataframe (pd.DataFrame): dataset that has been read into a dataframe

        Raises:
            ValueError: When dataframe is empty in value.

        Returns:
            None

        """
        # check if the dataframe has any rows
        if dataframe.shape[0] == 0:
            log_string = "DataFrame is empty"
            logging.error(log_string, stack_info=True)
            raise ValueError(log_string)
        # Check if the DataFrame only contains NaN values
        if dataframe.isna().all().all():
            log_string = "The DataFrame only contains NaN values."
            logging.error(log_string, stack_info=True)
            raise ValueError(log_string)

        return None

    def check_target_feature(self, dataframe_columns: list) -> None:
        """Check if target feature exists in the transaction and final combined dataframe. Raises a KeyError and terminates the program if target feature does not exist as this is a critical error.

        Args:
            dataframe (pd.DataFrame): transaction, merged dataframe

        Raises:
            KeyError: Target feature is missing from dataframe
        """
        if TARGET_FEATURE_NAME not in dataframe_columns:
            logging.error(f"Target feature is missing from dataframe")
            raise KeyError(f"Target feature is missing from dataframe")
        return None

    def check_file_format_encoding(self, filepath) -> None:
        """Check if the file's format and encoding conform to the accepted types.
        - csv or xlsx
        - utf-8 or utf-8-sig
        Modify csv files to follow the specified encoding type if they do not conform.
        For xlsx files, a log message is generated which requires the user to manually modify the encoding.

        Args:
            None

        Returns:
        True/False
            None
        """
        # check if file is csv, try to open file as file might have the correct extension, but is in an invalid format
        if filepath.endswith(".csv"):
            try:
                df = pd.read_csv(filepath)
            except UnicodeDecodeError:
                logging.error(f"Invalid file format")
            # check encoding type and change to specified encoding (utf-8) if it is not correct.
            with open(filepath, "rb") as f:
                content = f.read()
                if chardet.detect(content)["encoding"].lower() not in ENCODINGTYPE:
                    df.to_csv(filepath, encoding=ENCODINGTYPE[0])
        elif filepath.endswith(".xlsx"):
            try:
                workbook = openpyxl.load_workbook(filepath)
            except UnicodeDecodeError:
                logging.error(f"Invalid file format")
            # check encoding type
            for sheet in workbook.worksheets:
                if sheet.encoding.lower() not in ENCODINGTYPE:
                    logging.error(
                        f"encoding is not in the accepted format: {ENCODINGTYPE}"
                    )
        else:
            logging.error(f"filetype is not in the accepted format")
        return None
