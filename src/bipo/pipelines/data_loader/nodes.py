import pandas as pd
from typing import Dict, Union

from kedro.io import DataSetError, DataCatalog
from kedro.config import ConfigLoader

from bipo import settings
import logging
import re
from dateutil import parser

logger = logging.getLogger(settings.LOGGER_NAME)
conf_loader = ConfigLoader(conf_source=settings.CONF_SOURCE)
const_dict = conf_loader.get("constants*")
# DEFAULT CONSTANTS
DEFAULT_DATE_COL = const_dict["default_date_col"]


def merge_unique_daily_partitions(
    partitioned_input: Dict[str, pd.DataFrame]
) -> Union[None, pd.DataFrame]:
    """Function which merges all data partitions (dataframes) from 'unique_daily_records' subfolder on a date column (as reference from the global variable) with an instantiated dataframe containing a column containing date values which covers the date period of interest.

    This function also calls merge_unique_csv_xlsx_df function.

    Args:
        partitioned_input (Dict[str, pd.DataFrame]): A dictionary with partition ids as keys and dataframe as values.

    Raises:
        None.

    Returns:
        pd.DataFrame: Merged dataframe if no exception. Else None is returned when no partitioned data files are available.
    """

    # Create an new dataframe representing the date period of interest. This would be used for merging with other dataframes in data partitions
    combine_df = None

    # Apply merge for each dataframe in the data partition input
    if not partitioned_input:
        return None

    for partition_id, partition_df in partitioned_input.items():
        logger.info(f"Loading data: {partition_id}")

        # By default the dataframe loaded would not use any column as index.
        df = partition_df

        logger.info(f"Loaded data is of shape {df.shape}")
        # Apply standard renaming
        df.rename(
            columns=lambda x: rename_columns(x),
            inplace=True,
        )
        renamed_date_col = rename_columns(DEFAULT_DATE_COL)
        # Apply pd.to_datetime conversion
        df[renamed_date_col] = pd.to_datetime(df[renamed_date_col])
        if combine_df == None:
            logger.info(f"Using {partition_id} as base..")
            combine_df = df.copy()
            continue

        # Execute merging when is more than 1 file in the partition, otherwise left as it is.
        logger.info(f"Attempting to merge {partition_id}")

        # Check if the date column appears in incoming dataframe and existing dataframe
        if renamed_date_col in df:
            combine_df = merge_unique_csv_xlsx_df(df, combine_df)
            logger.info(f"After merging: dataframe is of shape {combine_df.shape}")
        else:
            logger.info("Skipping merge due to no common date columns")

    logger.info(f"Setting {renamed_date_col} as index")
    combine_df.set_index(renamed_date_col, inplace=True)
    return combine_df


def merge_unique_csv_xlsx_df(
    df_to_merge: pd.DataFrame, base_df: pd.DataFrame
) -> pd.DataFrame:
    """Function which merges two dataframe using right merge (df_to_merge to base_df) base on global DEFAULT_DATE_COL which represents the common date col of the dataframe. It assumes all dataframe columns are renamed to at least a lowercase.

    Args:
        df_to_merge (pd.DataFrame): DataFrame representing a specific dataset of interest
        base_df (pd.DataFrame): DataFrame representing a dataset containing mixture of datasets column which is to be merged onto.

    Raises:
        None.

    Returns:
        pd.DataFrame: Merged dataframe.
    """
    renamed_date_col = rename_columns(DEFAULT_DATE_COL)

    try:
        combine_df = pd.merge(
            df_to_merge,
            base_df,
            how="outer",
            left_on=renamed_date_col,
            right_on=renamed_date_col,
        )
    except KeyError:
        logger.info(f"Unable to merge on index due to differences.")
        combine_df = pd.merge(df_to_merge, base_df, how="cross")

    logger.info(f"Current merged dataframe {combine_df.shape}")
    return combine_df


def rename_merge_unique_csv_xlsx_df_col_index(df: pd.DataFrame) -> pd.DataFrame:
    """This function renames all columns of merged csv and xlsx data, and set the renamed date related column as dataframe index.

    Args:
        df: Dataframe to be processed.

    Raises:
        None.

    Returns:
        pd.DataFrame: Merged dataframe.

    """

    # Apply reset to ensure all column names are captured.
    df.reset_index(inplace=True)
    df.columns = [rename_columns(col) for col in df.columns]

    # Select the column which is of datetime
    datetime_col = df.select_dtypes(include=["datetime"]).columns.tolist()
    # Assume default date col is not renamed, we want to retain its original form
    if len(datetime_col):
        logger.info(
            "Setting identified first instance datetime column as index and standardise its naming"
        )
        df.set_index(datetime_col[0], inplace=True)
        # Rename date index to DEFAULT_DATE_COL instead as used commonly.
        df.index.names = [DEFAULT_DATE_COL]
    else:
        logger.info(f"No relevant date column found in dataframe.\n")

    return df


def load_and_partition_proxy_revenue_data(df: pd.DataFrame) -> Dict:
    """Function which partitions the proxy revenue data by its outlet column into smaller datasets.

    Args:
        df (pd.DataFrame): DataFrame containing daily outlet proxy revenue data.

    Raises:
        KeyError: When specified column to access is not found in dataframe.

    Returns:
        Dict: Dictionary containing filename and partitioned dataframe.
    """
    outlet_part_dict = {}
    logger.info("Partitioning outlet revenue data")

    # Read column name
    outlet_col = const_dict["default_outlet_column"]
    unique_outlet_list = df[outlet_col].unique()

    try:
        for outlet in unique_outlet_list:
            logger.info(f"Processing outlet: {outlet}")
            outlet_df = df[df[outlet_col] == outlet].copy()
            # Set date as index and save dataframe as a partition (where index is saved)
            outlet_df.set_index(DEFAULT_DATE_COL, inplace=True)
            logger.info(f"Renamed columns after partitioning: {outlet_df.columns}")
            logger.info(f"Propensity data is of shape {outlet_df.shape}")
            outlet_df.rename(
                columns=lambda x: rename_columns(x),
                inplace=True,
            )

            # .csv will be appended as defined in catalog.yml
            outlet_part_dict[f"proxy_revenue_{outlet}"] = outlet_df

        logger.info("Completed partitioning of all outlet revenue data.\n")
        return outlet_part_dict

    except KeyError:
        raise KeyError(f"Unable to access required column: {outlet_col}")


def load_and_structure_propensity_data(df: pd.DataFrame) -> pd.DataFrame:
    """Function which load and restructre propensity data by deriving daily aggregated mean using region-based propensity to spend factor.

    Args:
        df (pd.DataFrame): DataFrame containing daily region-based propensity to spend data.

    Raises:
        KeyError: When specified column to access is not found in dataframe.

    Returns:
        pd.DataFrame: Dataframe containing the mean-aggregated propensity to spend factor using all regional factors provided.
    """
    logger.info("Aggregating region based propensity data to region based form")
    # Get the name of the column representing propensity data
    factor_col = const_dict["default_propensity_factor_column"]

    try:
        df[factor_col] = df[factor_col].apply(pd.to_numeric, errors="coerce")

        # Take mean value of propensity data to represent as national level
        groupby_df = df.groupby(DEFAULT_DATE_COL)[[factor_col]].mean()

        logger.info(
            f"Renamed propensity data columns after groupby: {groupby_df.columns} with shape {groupby_df.shape}\n"
        )

        # Apply standard renaming
        groupby_df.rename(
            columns=lambda x: rename_columns(x),
            inplace=True,
        )

        return groupby_df

    except KeyError:
        raise KeyError(f"Unable to access required column: {factor_col}.\n")


def load_and_structure_marketing_data(df: pd.DataFrame) -> pd.DataFrame:
    """Function which load and restructre marketing data by converting it into a new dataset comprising daily-based marketing expenditure costs over respective marketing modes with names of active marketing campaign names

    Args:
        df (pd.DataFrame): Dataframe containing summarised marketing channel spends by campaigns.

    Raises:
        KeyError: When specified column to access is not found in dataframe.

    Returns:
        pd.DataFrame: Restructured dataframe containing daily-based marketing expenditure costs over respective marketing modes with names of active marketing campaign names.
    """
    # Apply pivoting for marketing data using campaign info, with total cost serving as values for various mode

    mkt_channel_col_name = const_dict["default_mkt_channels_column"]
    mkt_cost_col_name = const_dict["default_mkt_cost_column"]
    mkt_name_col = const_dict["default_mkt_name_column"]
    date_start, date_end = const_dict["default_mkt_date_start_end_columns"]
    mkt_channel_col_list = df[mkt_channel_col_name].unique()

    try:
        df_pivot = df.pivot(
            index=[mkt_name_col, date_start, date_end],
            columns=[mkt_channel_col_name],
            values=[mkt_cost_col_name],
        )

        # Drop campaigns where channel costs(mode) are all empty and fill nulls for remaining campaigns with some missing cost. Subsequently remove excess column level due to effects of pivoting
        df_pivot = df_pivot.dropna(how="all").fillna(0)
        df_pivot.columns = df_pivot.columns.droplevel(0)

        # Convert marketing data to daily equivalent form. Note that some dates have duplicates due to different campaigns
        df_pivot = df_pivot.reset_index().rename_axis(None, axis=1)
        df_pivot[DEFAULT_DATE_COL] = df_pivot.apply(
            lambda x: pd.date_range(start=x[date_start], end=x[date_end]), axis=1
        )
        df_pivot = df_pivot.explode(DEFAULT_DATE_COL)

        # Calculate duration between start and end and + 1 to include start date. subsequently adjust to daily average

        df_pivot["Duration"] = (df_pivot[date_end] - df_pivot[date_start]).dt.days + 1
        df_pivot[mkt_channel_col_list] = df_pivot[mkt_channel_col_list].div(
            df_pivot["Duration"], axis=0
        )

        # Generate a mkt list and date info df
        mkt_date_df = df_pivot.groupby(DEFAULT_DATE_COL)[mkt_name_col].agg(list)

        # Daily mkt channel cost for each day
        mkt_daily_cost_df = (
            df_pivot.drop(columns=mkt_name_col)
            .groupby(DEFAULT_DATE_COL)[mkt_channel_col_list]
            .sum()
        )

        # Merge the two created dataframe
        df = pd.merge(mkt_date_df, mkt_daily_cost_df, left_index=True, right_index=True)

        # Rename only cost related information with daily_cost suffix, to ensure naming consistency across and their meaning after value transformation
        df.rename(
            columns=lambda x: rename_columns(x) + "_daily_cost"
            if x != mkt_name_col
            else rename_columns(x),
            inplace=True,
        )

        logger.info(f"Renamed marketing columns: {df.columns} with shape {df.shape}\n.")
        return df

    except KeyError:
        raise KeyError(
            f"Unable to access required columns involving either of the columns: {mkt_name_col}, {date_start}, {date_end} ,{mkt_channel_col_name},{mkt_cost_col_name}"
        )


def load_and_structure_weather_data(df: pd.DataFrame) -> pd.DataFrame:
    """Function which applies mean aggregation of the region-based weather data involving temperature, rainfall and wind speeds. Other non-numerical features are dropped.

    Args:
        df (pd.DataFrame): DataFrame containing region-based weather related data

    Raises:
        KeyError: When specified column to access is not found in dataframe.

    Returns:
        pd.DataFrame: Dataframe containing the mean aggregated weather data.
    """
    # Consturct date using defined columns from constants.yml
    date_col_list = const_dict["columns_to_construct_date"]["weather_data"]
    try:
        logger.info(f"Constructing date from {date_col_list}")
        df[date_col_list] = df[date_col_list].astype(str)
        df[DEFAULT_DATE_COL] = pd.to_datetime(
            df[date_col_list].agg("-".join, axis=1), format="%Y-%m-%d"
        )
        logger.info(
            f"Removing {date_col_list} as {DEFAULT_DATE_COL} has been constructed.\n"
        )
        df.drop(columns=date_col_list, inplace=True)

    except KeyError:
        raise KeyError(
            f"Attempt to access dataframe features based on {date_col_list} but at least one is not in."
        )

    # Aggregate region using mean
    logger.info("Applying mean aggregation for all columns")

    # Filter out date related columns to facilitate groupby because of possible nans occurring for numerical columns.
    non_date_col = [
        col for col in df.columns if col != DEFAULT_DATE_COL and col != date_col_list
    ]

    # Coerce possible numeric features to numeric, else as it is
    df[non_date_col] = df[non_date_col].apply(pd.to_numeric, errors="coerce")

    # Only numerical columns would be mean aggregated, others would be nans due to object type, which would be dropped since we are only interested in numeric columns
    groupby_df = df.groupby(DEFAULT_DATE_COL, dropna=True)[non_date_col].mean()
    groupby_df.rename(
        columns=lambda x: rename_columns(x),
        inplace=True,
    )
    groupby_df.dropna(axis=1, inplace=True)

    logger.info(
        f"Renamed weather data columns after groupby: {groupby_df.columns} with shape {groupby_df.shape}\n."
    )

    return groupby_df


def rename_columns(string: str) -> str:
    """Function which rename a string by lowercasing, stripping trailing and leading spaces, replacing spaces inbetween words with _ and finally remove non-alphanumeric characters (except _).

    Example: ' Date #$Q(' becomes 'date_q'

    Args:
        string (str): Input string

    Raises:
        None

    Returns:
        str: Processed string if input is not None. Else default 'unknown' string is returned instead.
    """
    if isinstance(string, str):
        new_string = string.lower().strip().replace(" ", "_")
        return re.sub(r"[^a-zA-Z0-9_]", "", new_string)
    else:
        logger.info("Encountered other type, not processing.\n")
        return string
