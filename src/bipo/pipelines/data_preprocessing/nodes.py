import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Set, Union, Any
from datetime import datetime

from kedro.config import ConfigLoader
from bipo import settings
from kedro.io import DataSetError
import logging

conf_loader = ConfigLoader(conf_source=settings.CONF_SOURCE)
const_dict = conf_loader.get("constants*")

logger = logging.getLogger(settings.LOGGER_NAME)


def outlet_exclusion_list_check(outlet_exclusion_list: List) -> Set:
    """Function which conducts a type check on input argument outlet_exclusion_list.

    Args:
        outlet_exclusion_list (List): List containing representation outlets/cost centres in either string/numerical representation.

    Raises:
        None.

    Returns:
        Set: Python set of input list or a default set reference from constants.yml's 'default_outlets_exclusion_list' parameters config.
    """
    # Return default exclusion list if specified outlet_exclusion_list is not of list type.
    if not isinstance(outlet_exclusion_list, List):
        outlet_to_exclude_list = set(const_dict["default_outlets_exclusion_list"])
        logger.error(
            f"The provided outlet_exclusion_list from parameters.yml does not match expected list type. Using default setting: {outlet_to_exclude_list}"
        )
    else:
        outlet_to_exclude_list = set([str(outlet) for outlet in outlet_exclusion_list])
    return outlet_to_exclude_list


def const_value_perc_check(perc_value: float) -> float:
    """Function which conducts a percentage value check by type casting it into a valid float, clipping the value to the range between 0 and 100 (both values inclusive). Should float type casting fails, a default value referencing constants.yml is used.

    Args:
        perc_value (float): Floating value of percentage.

    Raises:
        None.

    Returns:
        float: Clipped input value to between 0 and 100, representing percentage validity.
    """

    # Attempt to type cast input to float before clipping it to valid range, 0 and 100
    try:
        perc_value_float = float(perc_value)
        perc_value_float = np.clip(perc_value_float, 0, 100)

    except ValueError:
        perc_value_float = const_dict["default_const_value_perc_threshold"]
        logger.error(
            f"Attempting to type cast {perc_value} to integer by failed. Using default values {perc_value_float}"
        )
    return perc_value_float


def date_validity_check(start_date: str, end_date: str) -> Tuple[str, str]:
    """Function which checks if the input start and end dates are in the correct format of '%Y-%m-%d' as well as its time ordering.

    Args:
        start_date (str): Date in string following '%Y-%m-%d' format
        end_date (str): Date in string following '%Y-%m-%d' format

    Raises:
        None

    Returns:
        Tuple[str, str]: Validated start and end dates.
    """
    # Apply reformat
    try:
        start_date = datetime.strptime(start_date, "%Y-%m-%d").strftime("%Y-%m-%d")
        end_date = datetime.strptime(end_date, "%Y-%m-%d").strftime("%Y-%m-%d")

        if start_date > end_date:
            logger.info("The start date is after the end date, reversing the dates")
            end_date, start_date = start_date, end_date

    except ValueError:
        logger.error(
            "Invalid format detected, using defaults configurations from constants.yml"
        )
        start_date = const_dict["default_start_date"]
        end_date = const_dict["default_end_date"]

        # Format
        start_date = datetime.strptime(start_date, "%Y-%m-%d").strftime("%Y-%m-%d")
        end_date = datetime.strptime(end_date, "%Y-%m-%d").strftime("%Y-%m-%d")

        logger.error(f"Default start date {start_date} and end date {end_date}")

    return start_date, end_date


def merge_non_proxy_revenue_data(
    partitioned_input: Dict[str, pd.DataFrame]
) -> Union[pd.DataFrame, None]:
    """Function which merges all data in '02_dataloader/non_revenue_partition' into a single dataframe. These data represents non proxy revenue related datasets.

    Args:
        partitioned_input (Dict[str, pd.DataFrame]): A dictionary with partition ids as keys and dataframe as values.

    Returns:
        pd.DataFrame: Merged dataframe based on date if input partitioned_input is not empty.
    """
    # Instantiate a dummy dataframe
    merged_df = None
    try:
        for partition_id, partition_load_func in partitioned_input.items():
            logger.info(f"Loading partition: {partition_id}")
            df = partition_load_func
            # Set first instance of partition data as base dataframe. Subsequently use it to merge with other incoming partition data
            if merged_df is None:
                logger.info(f"Using {partition_id} as base dataframe.")
                # Conversion of datetime to ensure during pd.merge, merge is made on correct type without causing NaN as values.
                merged_df = df.copy()
                continue

            # For additional files, we need to merge them with outer join, where join is based on index. Joining on index via outer joins, avoids the creation of additional suffixes columns caused by merge.
            logger.info(f"Preparing merge with partition: {partition_id}")

            # To ensure that merge will not result in Nan when done on datetime

            # Merge on common
            merged_df = pd.concat(
                [df, merged_df], axis=1, join="outer", ignore_index=False
            )
            logger.info(f"Merged partition: {partition_id} to existing dataframe.")

        logger.info(
            f"Completed merging of non proxy revenue dataframe. Dataframe is of shape {merged_df.shape}.\n"
        )
    # merged_df will still be none since no partition is available for processing
    except DataSetError:
        logger.error("No data is available in the partition. Not merging anything.\n")

    return merged_df


def merge_outlet_and_other_df_feature(
    outlet_partitioned_input: Dict[str, pd.DataFrame],
    daily_other_feature_df: pd.DataFrame,
    params_dict: Dict[str, Any],
) -> Dict[str, pd.DataFrame]:
    """Function which merges non-proxy revenue data with individual outlet proxy revenue data located in '02_dataloader' folder.

    Args:
        outlet_partitioned_input (Dict[str, pd.DataFrame]): A dictionary with partition ids as keys and dataframe as values.
        daily_other_feature_df (pd.DataFrame): Dataframe comprising of non-proxy revenue related features.
        params_dict (Dict[str, Any]): Dictionary referencing parameters.yml.

    Raises:
        None.

    Returns:
        Dict[str, pd.DataFrame]: Dictionary representing processed outlet partitions if applicable.
    """
    # Check emptyness of daily_other_feature_df. Return as per outlet_partitioned_input if there is no other daily feature dataframe to merge.
    if daily_other_feature_df.empty:
        return outlet_partitioned_input

    # Instantiate a empty dict for storing processed partition file names as key and corresponding processed dataframe as values
    outlet_with_features_dict = {}

    logger.info(
        "Extracting start_date and end_date for data filtering from parameters.yml."
    )
    start_date = str(params_dict["start_date"])
    end_date = str(params_dict["end_date"])
    outlet_to_exclude_list = params_dict["outlets_exclusion_list"]
    zero_val_threshold_perc = params_dict["zero_val_threshold_perc"]

    # Apply data sanity check.
    start_date, end_date = date_validity_check(start_date, end_date)

    outlet_to_exclude_set = outlet_exclusion_list_check(outlet_to_exclude_list)

    zero_val_threshold_perc = const_value_perc_check(zero_val_threshold_perc)

    logger.info(f"Start date: {start_date}")
    logger.info(f"End date: {end_date}")

    # Load outlet partition. First, cross check with outlet_to_exclude_list to skip such partitions when it is in the list. Next, upon loading check if the revenue related column of the outlet

    for outlet_partition_id, outlet_partition_load_func in sorted(
        outlet_partitioned_input.items()
    ):
        outlet_partition_info = outlet_partition_id.split("_")[-1]
        if outlet_partition_info in outlet_to_exclude_set:
            logger.info(f"Skipping {outlet_partition_info} as it is to be excluded..\n")
            continue

        logger.info(f"Loading {outlet_partition_id}")
        outlet_df = outlet_partition_load_func

        # Check if the proxy revenue counts exceeds defined const_val_threshold_perc. If so, exclude to avoid downstream equal frequency binning approach.
        revenue_col = const_dict["default_revenue_column"]
        logger.info(
            "Checking if instances of specific values exceeds a set threshold, which may cause equal frequency binning issue."
        )

        if 0 in outlet_df[revenue_col].value_counts(normalize=True):
            if outlet_df[revenue_col].value_counts(normalize=True)[0] >= (
                zero_val_threshold_perc / 100
            ):
                logger.info(
                    f"{outlet_partition_id} excluded as there exists value(s) which exceeds the defined threshold. Moving to the next.\n"
                )
                continue

        logger.info("Retrieving earliest/latest available dates for outlets")
        earliest_avail_date = outlet_df.index.min()
        latest_avail_date = outlet_df.index.max()
        if earliest_avail_date > start_date or latest_avail_date < end_date:
            logger.info(
                f"Earliest date: {earliest_avail_date}, Latest date: {latest_avail_date}"
            )
            logger.info(f"Skipping {outlet_partition_id} due to insufficient data")
            continue
        else:
            # Apply left join for non-proxy revenue on outlet's proxy revenue as the date index of outlet is what we are interested.
            outlet_df.index = pd.to_datetime(outlet_df.index, format="%Y-%m-%d")
            daily_other_feature_df.index = pd.to_datetime(
                daily_other_feature_df.index, format="%Y-%m-%d"
            )
            outlet_df = pd.merge(
                outlet_df,
                daily_other_feature_df,
                how="left",
                left_index=True,
                right_index=True,
            )

            logger.info(
                f"After merging features, shape of {outlet_partition_id} is {outlet_df.shape}."
            )

        # Ensure date index format is %Y-%m-%d
        outlet_df.index = pd.to_datetime(outlet_df.index, format="%Y-%m-%d")

        # Update dictionary as output with new partition string
        new_partition_string = f"{outlet_partition_id}_processed"
        outlet_with_features_dict[new_partition_string] = outlet_df
    logger.info(
        "Completed merging of outlet proxy revenue and external features datasets.\n"
    )
    return outlet_with_features_dict
