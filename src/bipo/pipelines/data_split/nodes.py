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
from datetime import datetime
import os
import logging

# This will get the active logger during run time
logging = logging.getLogger(__name__)

# DEFAULT SETTING
CONST_TRAIN_RATIO = 60
CONST_VAL_RATIO = 20
CONST_TEST_RATIO = 20


def load_concat_outlet_feature_engineered_data(
    paritioned_input: Dict[str, str]
) -> pd.DataFrame:
    """Function that loads and combines all partitioned datasets based on pipeline.py the inputs args which this func is called.

    Args:
        paritioned_input (Dict): Dictionary containing all files name as key and its loader function based on type info as value (based on catalog.yml) based on the named input mapped to catalog.yml

    Raises:
        None

    Returns:
        pd.DataFrame: Dataframe representing feature engineered data contents.
    """
    # Set empty dataframe to prepare for concat
    merged_outlet_df = pd.DataFrame()
    for partition_id, partition_load_func in sorted(paritioned_input.items()):
        if "merged_" in partition_id:
            partition_data = partition_load_func()
            merged_outlet_df = pd.concat(
                [merged_outlet_df, partition_data], join="outer"
            )
            logging.info(f"Concatenated {partition_id} to existing dataframe")
        else:
            logging.info(
                f"{partition_id} file not included in the concatenation process."
            )

    # Assume date is confirmed inside
    merged_outlet_df["date"] = pd.to_datetime(merged_outlet_df["date"])
    merged_outlet_df.sort_values(by=["date", "cost_centre_code"], inplace=True)

    merged_outlet_df.set_index("date", inplace=True)
    # print(merged_outlet_df.shape)
    logging.info(f"Merged data is of shape: {merged_outlet_df.shape}")

    return merged_outlet_df


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

    # Get unique dates which would be used for time splits
    unique_date_index_list = sorted(df.index.unique())

    # Calculate the total duration available (need to add 1 to include start date)
    days_diff = unique_date_index_list[-1] - unique_date_index_list[0]

    total_duration = days_diff.days + 1

    # Calculate the number of days equivalent
    train_days = round(train_ratio * total_duration / 100)
    logging.info(f"Total days used for training: {train_days}/{total_duration}")
    val_days = round(val_ratio * total_duration / 100)
    logging.info(f"Total days used for validation: {val_days}/{total_duration}")
    test_days = total_duration - train_days - val_days
    logging.info(f"Total days used for testing: {test_days}/{total_duration}")

    # Define empty dataframes with existing columns name
    train_engineered_df = pd.DataFrame(columns=df.columns)
    val_engineered_df = pd.DataFrame(columns=df.columns)
    test_engineered_df = pd.DataFrame(columns=df.columns)

    # Update train,val test dataframes. Calculation of days is as follows:
    # 3 days of training from 1/1/2021 would be 1/1/2021 to 3/1/2021 (both dates inclusive). Hence by doing a start + duration - 1 would achieve such results
    if train_days > 0:
        train_date_end = df.index.min() + pd.Timedelta(train_days - 1, unit="days")
        logging.info(f"Training start date: {df.index.min()}")
        logging.info(f"Training end date: {train_date_end}")
        # Subset by index
        train_engineered_df = df.loc[df.index.min() : train_date_end, :]

        # For cases of positive val and test size
        if val_days > 0:
            val_date_start = train_date_end + pd.Timedelta(1, unit="days")
            val_date_end = val_date_start + pd.Timedelta(val_days - 1, unit="days")

            logging.info(f"Validation start date: {val_date_start}")
            logging.info(f"Validation end date: {val_date_end}")
            # Subset by index
            val_engineered_df = df.loc[val_date_start:val_date_end, :]

        if test_days > 0:
            test_date_start = (
                train_date_end
                + pd.Timedelta(val_days, unit="days")
                + pd.Timedelta(1, unit="days")
            )

            logging.info(f"Testing start date: {test_date_start}")
            logging.info(f"Testing end date: {df.index.max()}")
            # Subset by index
            test_engineered_df = df.loc[test_date_start:, :]

    logging.info(f"Training dataset shape: {train_engineered_df.shape}")
    logging.info(f"Validation dataset shape: {val_engineered_df.shape}")
    logging.info(f"Testing dataset shape: {test_engineered_df.shape}")

    # Return all 3 dataframe in a key value pair as part of PartitionedDataSet
    return {
        "merged_outlet_training": train_engineered_df.reset_index(),
        "merged_outlet_validation": val_engineered_df.reset_index(),
        "merged_outlet_testing": test_engineered_df.reset_index(),
    }
