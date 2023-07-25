"""
This is a boilerplate pipeline 'data_split'
generated using Kedro 0.18.10
"""
import pandas as pd
import numpy as np
import argparse
import openpyxl
from kedro.extras.datasets.pandas import CSVDataSet
from sklearn.model_selection import TimeSeriesSplit
from typing import Union, List, Tuple, Dict
import os

# This will get the active logger during run time
import sys

sys.path.append("../../")
sys.dont_write_bytecode = True

import logging

# This will get the active logger during run time
logging = logging.getLogger(__name__)

# DEFAULT SETTING
CONST_TRAIN_RATIO = 60
CONST_VAL_RATIO = 20
CONST_TEST_RATIO = 20


def load_feature_engineered_data(data: pd.DataFrame) -> pd.DataFrame:
    """Function that loads in a dataframe based on Kedro pipeline declaration.

    Args:
        data (pd.DataFrame): Expected data is Pandas dataframe.

    Raises:
        None

    Returns:
        pd.DataFrame: Dataframe representing feature engineered data contents.
    """

    logging.info(f"Loading input files as {type(data)}")
    # df = CSVDataSet(filepath=input_filepath, load_args={"index_col": 0})
    # Lowercase column names and set date
    data.columns = data.columns.str.lower()

    if "date" in data.columns:
        logging.info("Setting date column as index")
        data = data.set_index("date")
    else:
        log_string = "Assuming date has been set as index."

    return data


def do_train_val_test_split(
    df: pd.DataFrame, config_dict: Dict
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Function that applies dataframe index split by reading Kedro config file parameters automatically.

    Args:
        df (pd.DataFrame): Feature engineered dataFrame to be train-val-test split.
        config_dict (dict): Config dict containing configurations for index split proportions.

    Raises:
        None

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] containing:
        - train_engineered_df: pd.DataFrame representing the training dataset
        - val_engineered_df : pd.DataFrame representing the validation dataset
        - test_engineered_df : pd.DataFrame representing the testing dataset

    """

    # Makes assumption that class dataframe instance is not empty. Emptyness would have been caught under dataloader module.

    # Extract train, val and test values and set to integer values from config
    # Sanity checks

    # Any decimal values will be disregarded with int typecasting
    train_ratio = int(config_dict["time_index_split"]["train"]["value"])
    val_ratio = int(config_dict["time_index_split"]["val"]["value"])
    test_ratio = 100 - train_ratio - val_ratio

    # Bool indicator variable for resetting train-val-test ratios
    is_reassign_ratio = False

    # Set indicator variable to True if test ratio is higher than sum of train/val
    if test_ratio >= (train_ratio + val_ratio):
        log_string = "Test ratio is higher than the sum of train and val ratio. Preset to 60/20/20 train/val/test split ratio."

        logging.info(log_string)

        is_reassign_ratio = True

    # Consider case where we take all data as training data.
    if test_ratio < 0 or (val_ratio > train_ratio):
        log_string = "Sum for training and validation exceeds 100 or validation ratio exceeds train ratio. Will apply default 60/20/20 train/val/test split ratio instead to prevent program from terminating here."

        is_reassign_ratio = True
        logging.info(log_string)

    # Check the need to reassign ratio based on last updated is_reassign_ratio. # Set default ratios if true
    if is_reassign_ratio:
        train_ratio = CONST_TRAIN_RATIO
        val_ratio = CONST_VAL_RATIO
        test_ratio = CONST_TEST_RATIO

    # Calculate proportion. No need test_size calculcation as it will be handled automatically based on train and validation info
    train_size = int(train_ratio * len(df) / 100)
    val_size = int(val_ratio * len(df) / 100)
    test_size = int(test_ratio * len(df) / 100)
    # Define empty dataframes
    train_engineered_df = pd.DataFrame(columns=df.columns)
    val_engineered_df = pd.DataFrame(columns=df.columns)
    test_engineered_df = pd.DataFrame(columns=df.columns)

    # Update train,val test dataframes
    if train_size > 0:
        train_engineered_df = df.iloc[:train_size, :]

        # For cases of positive val and test size
        if val_size > 0:
            val_engineered_df = df.iloc[train_size : train_size + val_size, :]

        if test_size > 0:
            test_engineered_df = df.iloc[train_size + val_size :, :]

    # Return all updates
    return train_engineered_df, val_engineered_df, test_engineered_df
