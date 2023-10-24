# Script that caters for lag feature generations as part of feature engineering module.
from typing import List, Dict
import pandas as pd
import logging
import numpy as np
from bipo import settings
logger = logging.getLogger(settings.LOGGER_NAME)


def create_simple_lags(df: pd.DataFrame, lag_periods_list: List) -> pd.DataFrame:
    """Function which generates simple lag based on a list of defined lag periods. It assumes the dataframe contains all columns required for lag feature generation.

    Args:
        df (pd.DataFrame): Time-indexed dataframe to be processed.
        lag_periods_list (list): List of lags to generated.

    Raises:
        None

    Returns:
        pd.DataFrame: Dataframe containing derived lagged values of all dataframe features.
    """
    try:
        logger.info("Applying simple lags")

        default_df_col = list(df.columns)

        # Create multiple lag features based on provided column
        lag_df = df.assign(
            **{
                f"lag_{lag}_{col}": df[col].shift(lag)
                for lag in lag_periods_list
                for col in df.columns
            }
        )

        logger.info(
            f"Lag values: {lag_periods_list} generated. Dropping rows with entire nulls and retaining only generated columns"
        )
        # Drop the columns initially provided
        lag_df.drop(columns=default_df_col, inplace=True)
        lag_df.dropna(axis=0, how="any", inplace=True)
        logger.info(f"Generated lag features dataframe of shape {lag_df.shape}")

        return lag_df

    except ValueError:
        logger.error(
            "Unable to calculate simple lag due to invalid values in the columns to be processed."
        )
        return None


def create_sma_lags(
    df: pd.DataFrame, shift_period: int, sma_window_period_list: List
) -> pd.DataFrame:
    """Creates a dataframe containing generated lagged simple (equal weighted)moving average parameterised by both 'shift_period' (lags to make) & 'sma_window_period_list'(window duration which moving average is to be calculated) of a dataframe instance of a class

    Args:
        df (pd.DataFrame): DataFrame containing columns which simple moving average needs to be generated.
        shift_period (int): Number of days to shift dataframe which is equal to the earliest lagged number of days for inference (difference between last inference date and last provided lagged date).
        sma_window_period_list (list): Size of window used for calculating simple moving average.

    Raises:
        None.

    Returns:
        pd.DataFrame: Dataframe containing derived lagged simple moving average of all dataframe features.
    """
    default_df_col = list(df.columns)

    logger.info(
        f"Calculating simple moving average over specified periods (rowwise) of {sma_window_period_list}"
    )
    try:
        sma_df = df.assign(
            **{
                f"lag_{shift_period}_sma_{window}_days_{col}": df.rolling(
                    window=window, win_type=None, min_periods=None, axis=0
                )
                .mean()
                .shift(periods=shift_period)
                for window in sma_window_period_list
                for col in df.columns
            }
        )
        logger.info(
            f"Simple moving average calculation completed.Applying shift of {shift_period} periods"
        )

        # Drop original column used for simple moving average sales generation
        logger.info("Completed simple moving average calculation.\n")

        # Drop the columns initially provided
        sma_df.drop(columns=default_df_col, inplace=True)
        sma_df.dropna(axis=0, how="any", inplace=True)
        return sma_df

    except ValueError:
        logger.error("Unable to calculate simple moving average due to invalid values.")

        return None


def create_lag_weekly_avg_sales(
    df: pd.DataFrame, shift_period: int, num_weeks_list: list
) -> pd.DataFrame:
    """Function which constructs a groupings on a given dataset and derives the average values of the grouping determined by num_weeks_list before applying shift/lags to the generated values

    Args:
        df (pd.Dataframe): Dataframe to be processed with datetime as index.
        shift_period (int): Number of days to shift dataframe which is equal to the earliest lagged number of days for inference (difference between last inference date and last provided lagged date).
        num_weeks_list (list): List of week values required for lag values generations.

    Raises:
        None

    Returns:
        pd.DataFrame: dataframe with the moving average of weekly sales
    """

    logger.info("Generating data groups starting from earliest date available.")
    new_group_column_name = "group"
    default_df_col = list(df.columns)
    # Update dataframe with a new column containing grouped information.
    # This is to facilitate grouped average sales based on fixed  days grouped feature generation and would not be tracked as part of feature engineering

    df = generate_data_groups(df=df, new_column_name=new_group_column_name)

    # Create a base dataframe using date index and group information from input dataframe. This is to facilitate subsequent merge with the dataframe weekly lag aggregated means below. Index will be date based
    date_group_df = df[[new_group_column_name]].copy()

    # Apply a groupby based on generated groups and get the mean value based on column of interest which is sales. Note that this results in group column as index
    time_grouped_avg_df = df.groupby(new_group_column_name).mean()

    logger.info(
        f"Generated mean aggregated numerical columns based on groupings which represents a continuous time periods with {df.shape}."
    )
    # Apply rolling mean over the time period groups (fixed at 7 days) and apply a unit period equivalent based on time period groupings (now measured as a week equivalent). I.e. shift down each group aggregated average by 1 week if compared with daily data.
    weekly_rolling_mean_df = time_grouped_avg_df.assign(
        **{
            f"lag_mean_{num_week}_week_{col}": time_grouped_avg_df.rolling(
                window=num_week
            )
            .mean()
            .shift(1)
            for num_week in num_weeks_list
            for col in df.columns
            if col != new_group_column_name  # Due to the existence of grouping
        }
    ).reset_index()

    logger.info("Merging weekly rolling mean with created base df")
    # Set join above dataframe with

    date_index_name = date_group_df.index.name
    weekly_average_merge_df = (
        date_group_df.reset_index()
        .merge(weekly_rolling_mean_df, how="inner", on=new_group_column_name)
        .set_index(date_index_name)
    )

    # Set dataframe index as datetime
    weekly_average_merge_df.index = pd.to_datetime(
        weekly_average_merge_df.index, format="%Y-%m-%d"
    )

    # Shift the dataframe to match the inference period, where the period to shift is the difference between shift_period and the number of days per group ie. 7 for weekly.
    period = shift_period - 7
    weekly_average_merge_df = weekly_average_merge_df.shift(period)

    default_df_col.append(new_group_column_name)
    weekly_average_merge_df.drop(columns=default_df_col, inplace=True)
    return weekly_average_merge_df


def generate_data_groups(
    df: pd.DataFrame,
    new_column_name: str,
) -> pd.DataFrame:
    """Helper function called by lag_avg_weekly_sales to generate group information for each date entry of the dataframe, starting from earlist entry. A additional column based on input new_column_name representing group number is added to the original dataframe.

    Args:
        df (pd.DataFrame): pd.DataFrame to be processed.
        new_column_name (str): Column name to use for generating group info.

    Raises:
        None.

    Returns:
        pd.DataFrame: Subset dataframe containing added group column. None if non-dataframe is passed or KeyError is encountered.
    """

    logger.info("Retrieving first index of dataframe...")
    start_index = df.index[0]
    logger.info(f"First index retrieved is: {start_index}")
    # Create filtered subset dataframe based on start index
    subset_df = df.loc[start_index:, :].copy()

    # Assign group label. Doing an integer division would ensure consecutive blocks of data are grouped as single unit.
    subset_df[new_column_name] = np.arange(len(subset_df)) // 7 + 1
    logger.debug("Generated data groups via helper function.")
    return subset_df


def merge_fold_and_generated_lag(
    fold_based_outlet_partitioned_data: Dict[str, pd.DataFrame],
    generated_lags_dict: Dict[str, pd.DataFrame],
) -> Dict[str, pd.DataFrame]:
    """Function which merges a dictionary containing calculated outlet lagged values into a dictionary containing fold based outlet partitioned dataframe.

    Args:
        fold_based_outlet_partitioned_data (Dict[str, pd.DataFrame): Dictionary containing fold based outlet partitioned dataframes which are distinguished by fold and outlet information in the dictionary key.
        generated_lags_dict (Dict[str, pd.DataFrame): PartitionedDataset dictionary containing outlet-based lagged features stored as Kedro PartitionedDataSet that is accessible via its lazy loading function.

    Returns:
        Dict[str, pd.DataFrame]: Dictionary containing updated dataframe with lagged features if applicable. Else return same as input.
    """

    # Loop through the dictionary and join the outlet dataframe with lag values with the fold-based outlet dataframes to create an updated dataframe with lagged features.
    if not generated_lags_dict:
        logger.info(
            "No lag features to merge, skipping merging of such features to outlet dataframes.\n"
        )
        return fold_based_outlet_partitioned_data

    for (
        fold_based_outlet_id,
        fold_based_outlet_df,
    ) in fold_based_outlet_partitioned_data.items():
        # Fold based id is of the form eg. training_fold1_expanding_window_param_90_305
        outlet_info = fold_based_outlet_id.split("_")[-1]

        if isinstance(fold_based_outlet_df, pd.DataFrame):
            fold_based_outlet_df = fold_based_outlet_df
        else:
            fold_based_outlet_df = fold_based_outlet_df()

        fold_based_outlet_df.index = pd.to_datetime(
            fold_based_outlet_df.index, format="%Y-%m-%d"
        )

        # Assume lag features dataframe filename are prefix with lag_
        generated_lag_df = generated_lags_dict[f"lag_{outlet_info}"]

        generated_lag_df.index = pd.to_datetime(
            generated_lag_df.index, format="%Y-%m-%d"
        )
        df = fold_based_outlet_df.merge(
            generated_lag_df,
            how="left",
            left_index=True,
            right_index=True,
            suffixes=("", "_y"),
        )

        # Drop excess column with suffixes _y as defined during merge
        columns_with_y_suffixes = [col for col in df.columns if col.endswith("_)y")]
        df.drop(columns=columns_with_y_suffixes, inplace=True)

        # Update lag col list at first instance. This is consistent across folds and outlets since the difference only lies in the time index, with all features the same.
        lag_col_list = [col for col in df.columns if col.startswith("lag_")]

        # Drop rows which are impacted by nans
        df.dropna(subset=lag_col_list, inplace=True, axis=0)

        logger.info(
            f"Successfully merged {fold_based_outlet_id} with shape: {df.shape}"
        )
        fold_based_outlet_partitioned_data[fold_based_outlet_id] = df

    logger.info(
        "Finished merging of lag features with main data folds. Returning generated lag features.\n"
    )
    return fold_based_outlet_partitioned_data
