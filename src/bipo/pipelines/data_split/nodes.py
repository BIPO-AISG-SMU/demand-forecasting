import pandas as pd
from typing import List, Tuple, Dict, Any
from datetime import datetime
import logging
from kedro.config import ConfigLoader
from bipo import settings

# This will get the active logger during run time
logger = logging.getLogger(settings.LOGGER_NAME)
conf_loader = ConfigLoader(conf_source=settings.CONF_SOURCE)
data_split_constants = conf_loader.get("constants*")["data_split"]


def handle_invalid_folds_split_approach(
    fold: int, split_approach: str
) -> Tuple[int, str]:
    """Helper function that checks for validity of input parameters involving folds and split approach.

    Fold value would be set to default values based on settings defined in base/constants.yml if invalid values are detected for various splits; otherwise defaults to 1 with simple split approach being used instead.

    Args:
        fold (int): Fold info provided.
        split_approach (str): Split approach name info provided.

    Raises:
        None.

    Returns:
        Tuple containing the following:
        - folds (int): Updated folds if applicable.
        - split_approach (str): Updated split approach if applicable.
    """
    logger.info("Checking for validness of provided fold information")
    if fold < 1 or not isinstance(fold, int):
        if split_approach == "expanding_window" or split_approach == "sliding_window":
            logger.info(
                f"Detected invalid folds. Setting to 3 as default for split approach: {split_approach}"
            )
            fold = data_split_constants["window_split_fold_default"]
        else:
            logger.info(
                f"Detected invalid folds. Setting to {data_split_constants['data_split_option_default']} and using {data_split_constants['simple_split_fold_default']} split approach."
            )
            fold = data_split_constants["simple_split_fold_default"]
            split_approach = data_split_constants["data_split_option_default"]

    return fold, split_approach


def handle_invalid_days_for_split(
    training_days: int,
    validation_days: int,
    testing_days: int,
) -> Tuple[int, int, int]:
    """Helper function that checks for validity of input parameters involving training_days, validation_days and testing_days.

    If either of them are negative. A default would be set for the respective arguments through this function.

    Args:
        training_days (int): Training days info provided.
        validation_days (int): Validation days info provided.
        testing_days (int): Testing days info provided.

    Raises:
        None.

    Returns:
        Tuple containing the following:
        - training_days (int): Updated training days if applicable.
        - validation_days (int): Updated validation days if applicable.
        - testing_days (int): Updated testing days if applicable.
    """

    if (
        not isinstance(training_days, int)
        or (training_days < 0)
        or (training_days is None)
    ):
        training_days = data_split_constants["training_days_default"]
        logger.error(f"Invalid training days overridden to {training_days} days.")
    if (
        not isinstance(validation_days, int)
        or (validation_days < 0)
        or (validation_days is None)
    ):
        validation_days = data_split_constants["validation_days_default"]
        logger.error(f"Invalid validation days overridden to {validation_days} days.")

    if (
        not isinstance(testing_days, int)
        or (testing_days < 0)
        or (testing_days is None)
    ):
        testing_days = data_split_constants["testing_days_default"]
        logger.error(f"Invalid testing days overridden to {testing_days} days.")

    return (training_days, validation_days, testing_days)


def validate_split_params(
    data_split_params: Dict[str, Any]
) -> Tuple[int, int, int, int, str]:
    """Function that extracts split-related parameters from provided data_split_params parameters and checks the validity of inputs. Any invalid inputs would be overwritten by functions which references constants.yml for overwrite values.

    Args:
        data_split_params (Dict[str, Any]): Dictionary containing data split parameters involving training, testing and validation days information.

    Raises:
        None.

    Returns:
        Tuple[int, int, int, int, str] containing validated parameters:
        - training_days,
        - validation_days,
        - testing_days,
        - folds,
        - split_approach
    """

    try:
        # Get split approach, train, testing and val days from config
        training_days = data_split_params["training_days"]
        testing_days = data_split_params["testing_days"]
        validation_days = data_split_params["validation_days"]
        split_approach = data_split_params["split_approach"]
        folds = data_split_params["folds"]

    except KeyError:
        logger.error(
            "Error in reading required parameters to calculate necessary split parameters. Applying default values."
        )

    # Logic check for parameters involving training, testing,validation, window_sliding_stride and window_expansion_days using function call
    logger.info("Conducting logic check on split parameters.")
    (
        training_days,
        validation_days,
        testing_days,
    ) = handle_invalid_days_for_split(
        training_days,
        validation_days,
        testing_days,
    )

    # logic check for fold sand split_approach using function call.
    folds, split_approach = handle_invalid_folds_split_approach(folds, split_approach)

    logger.info("Completed logic check on split parameters.")

    return training_days, validation_days, testing_days, folds, split_approach


def prepare_for_split(df: pd.DataFrame, data_split_params: Dict[str, Any]) -> Dict:
    """Function that constructs a dataframe consolidating input dataframe and data split parameters for further split processing handled by other functions.

    Args:
        df (pd.DataFrame): Dataframe to be processed for split.
        data_split_params (Dict[str, Any]): Dictionary containing validated data split parameters involving training, testing and validation days information.

    Raises:
        None.

    Returns:
        Dict[str, Any]: Dictionary containing keys and values as the key name implies:
        - "dataframe",
        - "start_date_list": start_date_list,
        - "latest_date": latest_date,
        - "training_days_list": training_days_list,
        - "testing_days": testing_days,
        - "validation_days": validation_days,
        - "split_approach": split_approach,
        - "window_param_days": window_param_days,
    """
    # Get time period and duration related information
    date_col = conf_loader.get("constants*")["default_date_col"]

    # Ensure that date is in index of dataframe before further processing
    if date_col in df.columns:
        logger.info("Converting date column into datetime format and setting as index")
        df.set_index(date_col, inplace=True)

    # Set index as datetime
    df.index = pd.to_datetime(df.index, format="%Y-%m-%d")

    # Extract the earliest/latest availability date & duration from data
    earliest_date, latest_date, total_duration = get_date_info(df_index=df.index)

    logger.info(
        f"Data information - Earliest date: {earliest_date}, Latest date: {latest_date}, Duration: {total_duration} days"
    )
    # Validate split params with function call
    (
        training_days,
        validation_days,
        testing_days,
        folds,
        split_approach,
    ) = validate_split_params(data_split_params=data_split_params)

    # Assign earliest date as the start_date as a start. Used in the for loop generation below.
    start_date = earliest_date

    # Define a list to store parameters related to folds which would be an entry in the returned dictionary
    training_days_list = []
    start_date_list = []

    # Apply checks on window_expansion and window_sliding days as extracted from parameters/data_split.yml. For simple split, these parameters are set to 0
    window_expansion_days = data_split_params["window_expansion_days"]
    window_sliding_stride_days = data_split_params["window_sliding_stride_days"]
    window_expansion_days, window_sliding_stride_days = check_window_parameters(
        window_expansion_days, window_sliding_stride_days, split_approach
    )

    # Use a common window variable to store window_expansion_days and window_sliding_stride_days since they are mutually exclusive for each split approach. This is used as an output parameter of the existing function
    if split_approach == "expanding_window":
        # Assign to common window variable
        window_param_days = window_expansion_days
    elif split_approach == "sliding_window":
        # Assign to common window variable
        window_param_days = window_sliding_stride_days
    else:
        window_param_days = 0  # Simple split case

    # Loop through the number of folds requested and derive the necessary time periods
    for fold in range(1, folds + 1):
        logger.info(f"Processing {fold} using {split_approach}")

        # Expanding window case
        if split_approach == "expanding_window":
            # First fold do not include any 'exapnsion'
            if fold > 1:
                training_days += window_param_days
            logger.info(
                f"Days for training data based on fold #{fold} is {training_days} days for {split_approach} approach."
            )

        # Sliding window case
        elif split_approach == "sliding_window":
            start_date = earliest_date + pd.Timedelta(
                (window_param_days * (fold - 1)), unit="days"
            )
            logger.info(
                f"Start date based on fold #{fold} is {start_date} for {split_approach} approach."
            )

        # Calculate total days needed for each fold. This is ensure folds to be generated are not truncated in the process.
        total_days_required = training_days + testing_days + validation_days

        # Check if required fold end date exceeds latest date from data. If so, terminate the loop process to guarantee proper fold duration.
        fold_end_date = start_date + pd.Timedelta(
            (total_days_required - 1), unit="days"
        )
        if fold_end_date > latest_date:
            logger.info(
                f"Required end date {fold_end_date} exceeds {latest_date} available from data. Current fold up to required #{folds} folds would not be implemented."
            )
            # Update total applicable folds
            folds = fold - 1
            break
        else:
            logger.info(
                f"The start date for model training under fold #{fold}: is {start_date}, training days is: {training_days}"
            )
            training_days_list.append(training_days)
            start_date_list.append(start_date)

    logger.info(f"Number of folds applicable: {folds}\n")
    # Define split parameters dict as return
    split_params_dict = {
        "dataframe": df,
        "start_date_list": start_date_list,
        "latest_date": latest_date,
        "training_days_list": training_days_list,
        "testing_days": testing_days,
        "validation_days": validation_days,
        "split_approach": split_approach,
        "window_param_days": window_param_days,
    }
    return split_params_dict


def do_time_based_data_split(
    split_params_dict: Dict[str, Any],
) -> Dict[str, pd.DataFrame]:
    """Function that implements a time-based data split using split_params_dict.

    Args:
        split_params_dict (Dict[str, Any]): Dictionary containing key-value information pertaining to data split parameters configured in parameters/data_split.yml or applied defaults (if necessary).

    Raises:
        None.

    Returns:
        Dict[str, pd.DataFrame] continaining the following if input is not empty:
        - train_dict: Dictionary containing file name and pd.DataFrame representing the training dataset
        - val_dict : Dictionary containing file name and validation dataset
        - test_dict : Dictionary containing file name and testing dataset dataframe

        Otherwise empty dict.
    """

    # Return if input is empty as that indicates the split approach in prev step is invalid.
    if not bool(split_params_dict):
        logger.info(
            "Skipping time-based data splitting as necessary split parameters are not found."
        )
        return {}

    # Unpack the dictionary values
    logger.info("Preparing data for split from dictionary input")
    df = split_params_dict["dataframe"]
    split_approach = split_params_dict["split_approach"]
    start_date_list = split_params_dict["start_date_list"]
    latest_date = split_params_dict["latest_date"]
    training_days_list = split_params_dict["training_days_list"]
    testing_days = split_params_dict["testing_days"]
    validation_days = split_params_dict["validation_days"]
    window_param_days = split_params_dict["window_param_days"]

    # Train,val,test Dictionary used as return under Kedro framework with empty key-values
    train_val_test_dict = {}

    logger.info(f"Getting training_days_list: {training_days_list}")
    logger.info(f"Getting start_date_list: {start_date_list}\n")

    # Loop through all training days
    for nth_fold, (training_days, start_date) in enumerate(
        zip(training_days_list, start_date_list), start=1
    ):
        # Define empty dataframes with existing columns name
        logger.info(f"Processing fold #{nth_fold}")

        train_date_end = start_date + pd.Timedelta(training_days - 1, unit="days")
        val_date_start = train_date_end + pd.Timedelta(1, unit="days")
        val_date_end = val_date_start + pd.Timedelta(validation_days - 1, unit="days")

        logger.info(
            f"Extract dates between {start_date} and {train_date_end} for fold #{nth_fold} training dataset which is {training_days} days"
        )
        train_engineered_df = df.loc[start_date:train_date_end, :]

        logger.info(f"Training dataset shape: {train_engineered_df.shape}")

        # Update dictionary
        train_val_test_dict[
            f"training_fold{nth_fold}_{split_approach}_param_{window_param_days}"
        ] = train_engineered_df

        # For the case when validation dataset is needed, we update accordingly. As a result, this affects when the test dates
        if validation_days > 0:
            # Start validation date after the end of validation date
            val_date_start = train_date_end + pd.Timedelta(1, unit="days")
            val_date_end = val_date_start + pd.Timedelta(
                validation_days - 1, unit="days"
            )
            # Start test date after the end of validation date

            logger.info(
                f"Extract dates between {val_date_start} and {val_date_end} for validation dataset which is {validation_days} days"
            )
            val_engineered_df = df.loc[val_date_start:val_date_end, :]

            logger.info(f"Validation dataset shape: {val_engineered_df.shape}")

            train_val_test_dict[
                f"validation_fold{nth_fold}_{split_approach}_param_{window_param_days}"
            ] = val_engineered_df

            test_date_start = val_date_end + pd.Timedelta(1, unit="days")

        else:
            # Start test date after the end of training date
            logger.info("Not extracting any data for validation case")
            test_date_start = train_date_end + pd.Timedelta(1, unit="days")

        # Test case
        test_date_end = test_date_start + pd.Timedelta(testing_days - 1, unit="days")

        # Filter if date end is date start for testing set
        if test_date_end > test_date_start:
            logger.info(
                f"Extract dates between {test_date_start} and {test_date_end} for testing dataset which is {testing_days} days.\n"
            )
            test_engineered_df = df.loc[test_date_start:test_date_end, :]
            logger.info(f"Testing dataset shape: {test_engineered_df.shape}\n")
            train_val_test_dict[
                f"testing_fold{nth_fold}_{split_approach}_param_{window_param_days}"
            ] = test_engineered_df
        else:
            logger.info("Not extracting any data for testing case.\n")

    # Return all 3 dataframe in a key value pair as part of PartitionedDataSet
    return train_val_test_dict


def get_date_info(df_index: pd.Index) -> Tuple[datetime, datetime, int]:
    """Function that retrieves the earliest, latest date and total duration between the 2 dates (both inclusive).

    Args:
        df_index (pd.Index): DataFrame index to process.

    Raises:
        None.

    Returns:
        Tuple[datetime, datetime, int]:
        - earliest_date: earliest availability date
        - latest_date: latest availability date
        - duration: Total days between earliest and latest date (both inclusive)

    """

    # Get a list of unique dates and extract earliest/latest date
    unique_dates_list = sorted(df_index.unique())
    earliest_date = unique_dates_list[0]
    latest_date = unique_dates_list[-1]

    total_duration = (latest_date - earliest_date).days + 1

    return earliest_date, latest_date, total_duration


def check_window_parameters(
    window_expansion_days: int, window_sliding_stride_days: int, split_approach: str
) -> Tuple[int, int]:
    """Function that implements checks on window-related parameters involved in time-based splits and rectifies using default values specified in constants.yml or fixed constants depending on the split_approach.

    Example: The window_expansion_days is only applicable for expanding_window split approach and not window_sliding_stride_days. Any values set for the latter would be zeroed. Similarly applies for the case of sliding_window split approach.

    Args:
        window_expansion_days (int): Input for expanding window's expansion days.
        window_sliding_stride_days (int): Input for sliding window's sliding stride days.
        split_approach (str): String representation of split_approach.

    Raises:
        None.

    Returns:
        Tuple[int, int]: Tuple containing containing corrected/defaulted window_expansion_days, window_sliding_stride_days
    """
    if split_approach == "expanding_window":
        if (
            not isinstance(window_expansion_days, int)
            or (window_expansion_days < 0)
            or not window_expansion_days
        ):
            window_expansion_days = data_split_constants[
                "window_expansion_days_default"
            ]
            logger.error(
                f"Overriding to default {window_expansion_days} days due to invalid window expansion days input"
            )
        window_sliding_stride_days = 0

    if split_approach == "sliding_window":
        if (
            not isinstance(window_sliding_stride_days, int)
            or (window_sliding_stride_days < 0)
            or not window_sliding_stride_days
        ):
            window_sliding_stride_days = data_split_constants[
                "window_sliding_stride_days_default"
            ]
            logger.error(
                f"Detected invalid window_sliding_stride_days days. Overriding to default {window_sliding_stride_days} days."
            )
        window_expansion_days = 0

    return window_expansion_days, window_sliding_stride_days
