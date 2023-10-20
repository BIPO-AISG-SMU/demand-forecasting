"""
This is a boilerplate pipeline 'pre_feature_engineering'
generated using Kedro 0.18.10
"""
import pandas as pd
from typing import Union, List, Tuple, Dict, Any
from datetime import datetime
from kedro.config import ConfigLoader
from bipo import settings
import logging
from kedro.config import ConfigLoader
from .feature_indicator_diff_creation import (
    create_min_max_feature_diff,
    create_is_weekday_feature,
    create_is_holiday_feature,
    create_is_raining_feature,
    create_is_pandemic_feature,
)
from bipo.utils import get_project_path

logger = logging.getLogger(settings.LOGGER_NAME)
conf_loader = ConfigLoader(conf_source=settings.CONF_SOURCE)
conf_const = conf_loader.get("constants*")


def create_bool_feature_and_differencing(
    partitioned_dict: Dict[str, pd.DataFrame], params_dict: Dict[str, Any]
) -> pd.DataFrame:
    """Function which drops a validated list of columns which is obtained by comparing between provided columns_to_drop_list against the dataframe columns information,

    Args:
        partitioned_input (Dict[str, pd.DataFrame]): Kedro MemoryDataSet dictionary containing outlet features dataframe to be processed.
        params_dict (Dict[str, Any]): Dictionary containing parameters referenced from parameters.yml
    Returns:
        Dict: Dictionary containing partition names of files and data.
    """
    # Extract parameters of interest.
    pandemic_column = params_dict["fe_pandemic_column"]
    holiday_column_list = params_dict["fe_holiday_column_list"]
    rainfall_column = params_dict["fe_rainfall_column"]
    min_max_feature_list = params_dict["columns_to_diff_list"]
    mkt_column = params_dict["fe_mkt_column_name"]

    partition_output_dict = {}
    for partition_id, df in partitioned_dict.items():
        df.index = pd.to_datetime(df.index, format="%Y-%m-%d")
        logger.info("Generating weekday indicator feature.")

        df = create_is_weekday_feature(df)
        if min_max_feature_list:
            logger.info(
                "Generating differencing features using supplied min,max columns."
            )
            df = create_min_max_feature_diff(
                df=df, min_max_column_list=min_max_feature_list
            )
        if holiday_column_list:
            logger.info("Generating holiday indicator feature.")
            df = create_is_holiday_feature(
                df=df, holiday_type_col_list=holiday_column_list
            )
        if rainfall_column:
            logger.info("Generating rainfall indicator feature.")
            df = create_is_raining_feature(df=df, rainfall_col=rainfall_column)

        if pandemic_column:
            logger.info("Generating pandemic indicator feature.")
            df = create_is_pandemic_feature(df, pandemic_col=pandemic_column)

        logger.info("Completed necessary feature indicator/differencing features")
        partition_output_dict[partition_id] = df
    return partition_output_dict


def create_mkt_campaign_counts_start_end(
    df: pd.DataFrame, params_dict: Dict[str, Any]
) -> pd.DataFrame:
    """Function which generates new boolean indicators for days which new marketing campaigns occur, as well as the the final day of the existing campaign and the number of active campaign ongoing.

    Args:
        df (pd.DataFrame): Marketing dataframe to be processed.
        params_dict (Dict[str, Any]): Dictionary containing parameters referenced from parameters.yml.

    Raises:
        None

    Returns:
        Tuple containing:
        - pd.DataFrame: Updated marketing dataframe with generated boolean column.
    """
    mkt_campaign_col = params_dict["fe_mkt_column_name"]
    logger.info(f"Processing marketing campaign name column: {mkt_campaign_col}")

    # New column name identifiers
    mkt_count_col_name = f"{mkt_campaign_col}_counts"
    mkt_start = f"is_{mkt_campaign_col}_start"
    mkt_end = f"is_{mkt_campaign_col}_end"

    # Get date period of interest
    start_date = params_dict["start_date"]
    end_date = params_dict["end_date"]

    # Get date column from constants.yml
    default_date_col = conf_const["default_date_col"]

    # Convert new datetimeindex to dataframe. Drop excess column when converting datetime index to dataframe
    new_df = pd.date_range(
        start=start_date,
        end=end_date,
        freq="D",
    ).to_frame(name=default_date_col)

    # Ensure index is converted to datetime
    new_df.index = pd.to_datetime(new_df.index, format="%Y-%m-%d")
    df.index = pd.to_datetime(df.index, format="%Y-%m-%d")

    # Use new_df as a base for df to be joined to based on index, which is time
    df = new_df.join(df)
    if mkt_campaign_col in df.columns:
        # Fill null with None before deriving number of campaigns as listed in the form of [Campaign1,Campaign2].Splitting a string without ',' results in itself.
        df[mkt_count_col_name] = (
            df[mkt_campaign_col]
            .fillna("None")
            .apply(lambda x: len(x.split(",")) if x else 0)
        )

        # Compare number of ongoing campaigns of between the day and the day before to identify if there is an increase in length which signifies new campaign
        df[mkt_start] = df[mkt_count_col_name].diff() > 0
        # Due to differencing, we need to impute first entry. Set to 0 (false) by assuming that an event is unlikely to start on the first entry of the data.
        df[mkt_start] = (
            pd.to_numeric(df[mkt_start], errors="coerce").fillna(0).astype(int)
        )

        # Compare number of ongoing campaigns of between the day and the day before to identify if there is an decrease in length which signifies new campaign.

        df[mkt_end] = df[mkt_count_col_name].diff() < 0
        df[mkt_end] = pd.to_numeric(df[mkt_end], errors="coerce").astype(int)

        # Due to differencing effect for ending of campaign,we need to shift the values up since calculation is applied on later index instead of previous time index. This results in last entry having null. Set to 0 (false) by assuming that an event is unlikely to end on the last entry of the data.
        df[mkt_end] = df[mkt_end].shift(-1).fillna(0)

        logger.info(
            f"Created a indicator features related to start/end for {mkt_campaign_col}"
        )

    else:
        # Assume marketing counts, start and end of campaigns indicator are 0 if wrong column is referenced.
        logger.error(
            f"Unable to create a boolean indicator for start/end of a marketing event based on given column: {mkt_campaign_col} as it does not exist.\n"
        )
        data = {mkt_count_col_name: 0, mkt_start: 0, mkt_end: 0}
        df = df.assign(**data)
    return df[[mkt_count_col_name, mkt_start, mkt_end]]


def drop_columns(
    partitioned_input: Dict[str, pd.DataFrame], params_dict: Dict[str, Any]
) -> Dict[str, pd.DataFrame]:
    """Function which drops a validated list of columns which is obtained by comparing between provided columns_to_drop_list against the dataframe columns information. This is the first function that is called as part of time_agnostic_feature_engineering module process.

    Args:
        partitioned_input (Dict[str,  pd.DataFrame]): Kedro MemoryDataSet dictionary containing outlet features dataframe to be processed.
        params_dict (Dict[str, Any]): Dictionary containing parameters referenced from parameters.yml

    Raises:
        None

    Returns:
        Dict[str, pd.DataFrame]: Dictionary containing partition names of files and data.
    """
    # Otherwise, identify what are the columns that resides in the dataframe through set intesection with dataframe column
    col_to_drop_set = set(params_dict["fe_columns_to_drop_list"])

    if not col_to_drop_set:
        logger.info("No columns specified to drop. Skipping dropping process.\n")
        return {id: load_func for (id, load_func) in partitioned_input.items()}

    partition_output_dict = {}
    for partition_id, partition_load_df in partitioned_input.items():
        df = partition_load_df
        logger.info("Attempting to drop specified columns from dataframe.")
        # Case when columns to drop is empty

        df_columns_set = set(df.columns)

        valid_columns_list = list(col_to_drop_set.intersection(df_columns_set))

        df.drop(columns=valid_columns_list, inplace=True)
        logger.info(f"Dropped validated columns {valid_columns_list}.\n")

        partition_output_dict[partition_id] = df
    return partition_output_dict


def merge_fold_partitions_and_gen_mkt_data(
    partitioned_dict: Dict[str, pd.DataFrame], gen_mkt_df: pd.DataFrame
) -> Dict[str, pd.DataFrame]:
    """Function which left joins existing fold partitions and generated marketing dataframe features if exist. Otherwise, return the input partitioned_dict parameter.

    Args:
        partitioned_input (Dict[str, pd.DataFrame]): Kedro IncrementalDataSet dictionary containing individual outlet related features dataframe as values with filename as identifier.
        gen_mkt_df (pd.DataFrame): Pandas dataframe containing corrected marketing dataframe

    Raises:
        None

    Returns:
        Dict[str, pd.DataFrame]: Dictionary containing updated partitions data with unchanged paritioned ids.
    """
    if not gen_mkt_df.empty:
        for partition_id, df in partitioned_dict.items():
            logger.info("Joining marketing data to existing dataframe folds.")

            df.index = pd.to_datetime(df.index, format="%Y-%m-%d")

            gen_mkt_df.index = pd.to_datetime(gen_mkt_df.index, format="%Y-%m-%d")

            df = df.merge(gen_mkt_df, how="left", left_index=True, right_index=True)
            partitioned_dict[partition_id] = df

    return partitioned_dict


def no_mkt_days_imputation(
    partitioned_dict: Dict[str, pd.DataFrame], params_dict: Dict[str, Any]
) -> Dict[str, pd.DataFrame]:
    """Function which imputes a defined value imputation for days where no marketing related activities exist.

    Args:
        partitioned_input (Dict[str, pd.DataFrame]): Kedro MemoryDataSet dictionary containing keys with input partition names of files and dataframe with raw marketing columns.
        params_dict (Dict[str, Any]): Dictionary containing parameters referenced from parameters.yml

    Raises:
        None

    Returns:
        Dict[str, pd.DataFrame]: Dictionary containing keys with input partition names of files and dataframe with marketing cost related columns imputed.
    """
    impute_dict = params_dict["mkt_columns_to_impute_dict"]
    # Case when impute dict is not provided
    if not impute_dict:
        logger.info("Imputation dict provided is empty. Skipping this process.")
        return {id: load_func for (id, load_func) in partitioned_input.items()}

    mkt_column_name = params_dict["fe_mkt_column_name"]
    partition_output_dict = {}
    # Impute if dictionary is not empty.
    for partition_id, df in partitioned_dict.items():
        logger.info(
            f"Imputing features based on provided imputation dictionary: {impute_dict}"
        )
        try:
            # Impute cost values
            df.fillna(impute_dict, inplace=True)

            # Drop original mkt column, assuming it still exists
            df.drop(columns=mkt_column_name, inplace=True)
            partition_output_dict[partition_id] = df

        except KeyError:
            logger.error(
                "Unable to impute with the provided keys into the dataframe. Some features might not be imputed.\n"
            )

    return partition_output_dict


def segregate_outlet_based_train_val_test_folds(
    partitioned_dict: Dict[str, pd.DataFrame], params_dict: Dict[str, Any]
) -> Tuple[Dict, Dict, Dict]:
    """Function which takes in partitioned_dict containing respective folds of training/validation/testing consolidated outlets and segregates them by retraining/validation/testing subfolders followed by outlet.

    Args:
        partitioned_input (Dict[str, pd.DataFrame]): Kedro MemoryDataSet dictionary containing fold-based consolidated outlets.
        params_dict (Dict[str, Any]): Dictionary containing parameters referenced from parameters.yml.

    Raises:
        None

    Returns:
        Tuple[Dict, Dict, Dict]: Tuple of dictionary representing training and validation and testing dictionaries respectively, which is tied to Kedro Partitiondataset type
    """
    # Set empty dictionary to store train/val/test dictionaries.
    training_dict = {}
    validation_dict = {}
    testing_dict = {}

    # Identify the column representing outlet information
    outlet_column_name = params_dict["outlet_column_name"]

    logger.info(
        "Extracting relevant split and reorganising them into folders based on their file prefixes"
    )
    for partition_id, df in sorted(partitioned_dict.items()):
        outlet_list = list(df[outlet_column_name].unique())
        temp_dict = {}
        for outlet in outlet_list:
            outlet_df = df.loc[df[outlet_column_name] == outlet]
            new_partition_id = f"{partition_id}_{outlet}"
            temp_dict[new_partition_id] = outlet_df

        # Training fold assignment
        if partition_id.startswith("training"):
            training_dict.update(temp_dict)

        # Validation fold assignment
        elif partition_id.startswith("validation"):
            validation_dict.update(temp_dict)

        # Testing fold assignment
        elif partition_id.startswith("testing"):
            testing_dict.update(temp_dict)

        else:
            logger.info(
                f"Skipping {partition_id} as it contains unrecognisable prefix for assignment."
            )
            continue

        outlet_groupby_df = [
            groupby_df.reset_index(drop=True)
            for _, groupby_df in df.groupby([outlet_column_name])
        ]

    logger.info("Segregation by partition prefix completed.\n")
    return (
        training_dict,
        validation_dict,
        testing_dict,
    )
