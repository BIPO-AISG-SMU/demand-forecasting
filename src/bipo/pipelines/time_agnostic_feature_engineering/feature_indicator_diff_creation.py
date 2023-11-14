from typing import List
import pandas as pd
import logging
from bipo import settings

logger = logging.getLogger(settings.LOGGER_NAME)


def create_min_max_feature_diff(
    df: pd.DataFrame, min_max_column_list: List[List]
) -> pd.DataFrame:
    """Function which generates columns of differenced paired columns from a input list of list ('min_max_column_list') on a given dataframe ('df'). Original column pairs specified in min_max_column_list would NOT be dropped.

    Args:
        df (pd.DataFrame): Dataframe to processed.
        min_max_column_list (List[List]): List of list containing dataframe.paired columns representing the min and max values of a specific feature in such order.

    Raises:
        None.

    Returns:
        pd.DataFrame: Dataframe with generated differenced values features.
    """

    # Apply validation check on list elements in the input parameters min_max_column_list is of length 2(which signifies the min and max column belonging to a specific feature), as well as if the pairings belong to dataframe columns which is to be processed.

    validated_min_max_column_list = [
        sublist
        for sublist in min_max_column_list
        if len(sublist) == 2 and set(sublist).issubset(set(list(df.columns)))
    ]

    try:
        # Create new feature name to represent the max-min value as return value
        df = df.assign(
            **{
                f"diff_{min_col}_{max_col}": df[max_col] - df[min_col]
                for (min_col, max_col) in validated_min_max_column_list
            }
        )

    except ValueError:
        log_string = f"Unable to execute numeric operation provided column {column_name_min} or {column_name_max} as it not either a float/int column."
        logger.error(log_string, stack_info=True)

    return df


def create_is_weekday_feature(df: pd.DataFrame) -> pd.DataFrame:
    """Function which indicates if a day is either a weekday or weekend by setting it to true if it is a weekday, otherwise false based on a given dataframe index.

    Args:
        df (pd.DataFrame): Dataframe to be processed.

    Raises:
        None.

    Returns:
        pd.DataFrame: Updated dataframe with generated boolean representation column.
    """
    logger.info(
        "Creating a new indicator boolean column name 'is_weekday' to represent if date is a weekday/weekend"
    )
    if isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, format="%Y-%m-%d")
        df["is_weekday"] = df.index.weekday < 5
        df["is_weekday"] = pd.to_numeric(df["is_weekday"], errors="coerce").astype(int)
        logger.info(f"Created a boolean representations with is_weekday.")
    else:
        logger.info(
            "The dataframe index is not of datetime format, hence weekeday/weekend informaton cannot be derived. Skipping such process."
        )

    return df


def create_is_holiday_feature(
    df: pd.DataFrame, holiday_type_col_list: List
) -> pd.DataFrame:
    """Function which indicates if a day is either a weekday or weekend by setting it to true if it is a weekday, otherwise false based on a given dataframe index. Original column used in holiday_type_col_list would not be dropped.

    Args:
        df (pd.DataFrame): Dataframe to be processed.
        holiday_type_col_list (List): Specified list of holiday columns (public/school holiday) for processing.

    Raises:
        None.

    Returns:
        pd.DataFrame: Updated dataframe with generated boolean representation column for holiday feature.
    """
    logger.info("Generating boolean states for holiday related columns.")
    # Check the intersection of provided column list and dataframe columns. This is to ensure only valid columns are to be processed.
    valid_col_list = list(set(holiday_type_col_list).intersection(list(df.columns)))

    # Generate a list indicating new column names to be constructed representing the boolean equivalent status
    bool_col_list = [f"is_{col}" for col in valid_col_list]

    if valid_col_list:
        logger.info(f"Processing valid identified columns: {valid_col_list}")
        df[bool_col_list] = df[valid_col_list].notna()
        df[bool_col_list] = df[bool_col_list].astype(int)
        logger.info(f"Created a boolean representations with {bool_col_list}.\n")

    else:
        logger.info(
            "The provided holiday_column_list parameters did not match any columns in dataframe to be processed. Skipping such processing\n."
        )

    return df


def create_is_raining_feature(df: pd.DataFrame, rainfall_col: str) -> pd.DataFrame:
    """Function which indicates if a day has rained using 'rainfall_col' parameter for a given dataset 'df' using conditional evaluation with values > 0.2 or not (as defined by http://www.weather.gov.sg/climate-climate-of-singapore/). The rainfall_col column will not be dropped.

    Args:
        df (pd.DataFrame): Dataframe to be processed.
        rainfall_col (str): Specified rainfall column to reference when processing dataframe.

    Raises:
        None

    Returns:
        pd.DataFrame: Updated dataframe with generated boolean representation column.
    """
    if rainfall_col in df.columns:
        logger.info(f"Creating a boolean representation of {rainfall_col}")
        is_rainfall_col = f"is_{rainfall_col}"
        df[is_rainfall_col] = df[rainfall_col] > 0.2

        # Convert to int instead of bool
        df[is_rainfall_col] = pd.to_numeric(
            df[is_rainfall_col], errors="coerce"
        ).astype(int)

        logger.info(f"Created a boolean representation for {is_rainfall_col}.\n")
    else:
        logger.info(f"Unable to create a boolean representation of {rainfall_col}.\n")

    return df


def create_is_pandemic_feature(df: pd.DataFrame, pandemic_col: str) -> pd.DataFrame:
    """Function which indicates if there is pandemic restrictions based on input pandemic_col string parameter. The pandemic_col column will not be dropped.
    Args:
        df (pd.DataFrame): Dataframe to be processed.
        pandemic_col (str): Specified pandemic column in dataframe to reference when processing dataframe.

    Raises:
        None

    Returns:
        pd.DataFrame: Updated dataframe with generated boolean representation column.
    """
    if pandemic_col in df.columns:
        logger.info(f"Creating a boolean representation of {pandemic_col}")

        # New column name
        is_pandemic_col = "is_pandemic_restrictions"

        # If value is 'no limit', map such value to 0, otherwise 1 .
        df[is_pandemic_col] = df[pandemic_col].map(
            lambda x: 0 if x == "no limit" else 1
        )

        logger.info(f"Created a boolean representation for {is_pandemic_col}.\n")
    else:
        logger.info(f"Unable to create a boolean representation of {pandemic_col}.\n")

    return df


def create_marketing_counts_start_end_features(
    df: pd.DataFrame,
    mkt_campaign_col: str,
    mkt_count_col_name: str,
    mkt_start: str,
    mkt_end: str,
):
    """Function which creates marketing start and marketing end indicator features based on provided 'mkt_campaign_col' feature info.

    Args:
        df (pd.DataFrame): Dataframe to be processed.
        mkt_campaign_col (str): Specified marketing campaign column in dataframe to reference when processing dataframe.
        mkt_count_col_name (str): Column name containing number of marketing events that are active.
        mkt_start (str): Column name for an indicator feature representing start of a marketing event.
        mkt_end (str): Column name for an indicator feature representing end of a marketing event.

    Raises:
        None

    Returns:
        pd.DataFrame: Updated dataframe with generated mkt_count_col_name,mkt_start and mkt_end columns.
    """
    if mkt_campaign_col in df.columns:
        # Strip ",[,] symbols from marketing campaign col to facilitate event substring comparisons.
        df[mkt_campaign_col] = df[mkt_campaign_col].map(
            lambda x: x.strip('[]"') if not pd.isna(x) else ""
        )

        # Create a new features representing a up/down shift of marketing campaign col. This is used as part of condition involved in deciding if a date contains the start/end of a marketing event.
        df[f"{mkt_campaign_col}_shift_down"] = (
            df[mkt_campaign_col].shift().bfill(axis="rows")
        )
        df[f"{mkt_campaign_col}_shift_up"] = (
            df[mkt_campaign_col].shift(-1).ffill(axis="rows")
        )
        # Check if an event also exist in previous time period through column comparison.
        df["is_name_subset_as_prev"] = df.apply(
            lambda x: x[mkt_campaign_col] in x[f"{mkt_campaign_col}_shift_down"],
            axis=1,
        )

        # Get number of events for each day. Defaults to 0
        df[mkt_count_col_name] = (
            df[mkt_campaign_col]
            .fillna("None")
            .apply(lambda x: len(x.split(",")) if x else 0)
        )

        # Compare number of ongoing campaigns of between the day and the day before. For first entry without difference, treat as no change assuming event is a continuation.
        df["mkt_count_diff_prev"] = df[mkt_count_col_name].diff(1)
        df["mkt_count_diff_prev"].fillna(0, inplace=True)

        # Construct condition for marketing start where 1) the day must have at least an event with its event name not being part of any events in previous time period; OR 2), if there is positive value when comparing total events occurring based on mkt_count_diff_prev col.
        mkt_start_condition = (
            (df[mkt_count_col_name] >= 1) & (df["is_name_subset_as_prev"] == False)
        ) | (df["mkt_count_diff_prev"] > 0)

        # Apply condition and set value to numeric.
        df[mkt_start] = mkt_start_condition

        # Similar to the case right above,
        df["mkt_count_diff"] = df[mkt_count_col_name].diff()
        df["mkt_count_diff"].fillna(0, inplace=True)

        # Compare number of ongoing campaigns of between the day and the day before to identify if there is an increase in length which signifies new campaign AND there must be change in the marketing events. Alternatively, the increase in total events count as well indicates a start of new marketing event

        # Construct condition for marketing start where the day must have at least an event with its event name not being part of any events in previous time period OR, if there is positive value when comparing total events occurring based on mkt_count_diff_next col.
        mkt_start_condition = (
            (df[mkt_count_col_name] >= 1) & (df["is_name_subset_as_prev"] == False)
        ) | (df["mkt_count_diff_prev"] > 0)

        # Apply condition and set value to numeric.
        df[mkt_start] = mkt_start_condition

        df[mkt_start] = (
            pd.to_numeric(df[mkt_start], errors="coerce").fillna(0).astype(int)
        )
        # Compare number of ongoing campaigns of between the day and the day next. For last entry without difference, treat as no change assuming event continues.
        df["mkt_count_diff_next"] = df[mkt_count_col_name].diff(-1)
        df["mkt_count_diff_next"].fillna(0, inplace=True)

        # Check if same event occurs for next day. For no event,
        df["is_name_subset_as_next"] = df.apply(
            lambda x: x[mkt_campaign_col] in x[f"{mkt_campaign_col}_shift_up"],
            axis=1,
        )

        # For marketing end condition, check that 1) there is at least a marketing event AND the marketing event next day is not the same as current day; OR 2) that the next day total marketing event is lesser.
        mkt_end_condition = (
            (df[mkt_count_col_name] >= 1) & (df["is_name_subset_as_next"] == False)
        ) | (df["mkt_count_diff_next"] > 0)
        df[mkt_end] = mkt_end_condition
        df[mkt_end] = pd.to_numeric(df[mkt_end], errors="coerce").fillna(0).astype(int)

        logger.info(
            f"Created a indicator features related to start/end for {mkt_campaign_col}"
        )

        # Drop created temporary columns

        df.drop(
            columns=[
                "is_name_subset_as_prev",
                "is_name_subset_as_next",
                "mkt_count_diff_prev",
                "mkt_count_diff_next",
                f"{mkt_campaign_col}_shift_up",
                f"{mkt_campaign_col}_shift_down",
            ],
            inplace=True,
        )

    else:
        # Simply assume marketing counts, start and end of campaigns indicator are 0 if column required is wrongly referenced, i.e assume no marketing events info at all.
        logger.error(
            f"Unable to create a boolean indicator for start/end of a marketing event based on given column: {mkt_campaign_col} as it does not exist.\n"
        )
        data = {mkt_count_col_name: 0, mkt_start: 0, mkt_end: 0}
        df = df.assign(**data)
    return df[[mkt_count_col_name, mkt_start, mkt_end]]
