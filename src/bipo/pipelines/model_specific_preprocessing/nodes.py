"""
This is a boilerplate pipeline 'model_specific_preprocessing'
generated using Kedro 0.18.11
"""
import itertools
import pandas as pd
import logging
from typing import List, Dict, Any, Tuple
from typing import OrderedDict as OrdDict
from kedro.config import ConfigLoader
from collections import OrderedDict
from bipo import settings

# This will get the active logger during run time
logger = logging.getLogger(settings.LOGGER_NAME)
conf_loader = ConfigLoader(conf_source=settings.CONF_SOURCE)
const_dict = conf_loader.get("constants*")

# DEFAULT CONSTANTS
DEFAULT_DATE_COL = const_dict["default_date_col"]


def remove_unnecessary_columns_and_rows(
    partitioned_data: Dict[str, pd.DataFrame], params_dict: Dict[str, Any]
):
    """Function which drops features identified by parameters names with prefix fe in parameters.yml which is accessed through params_dict.

    Args:
        partitioned_data (Dict[str, pd.DataFrame]): A dictionary with partition ids as keys and dataframe as values.
        params_dict (Dict[str, Any]): Dictionary referencing parameters.yml.

    Raises:
        None.

    Returns:
        Dict[str, pd.DataFrame]: A dictionary similar to input partitioned_data with identified columns and rows removed.
    """

    # Identify parameters with 'fe' prefixes and outlet_column_name
    fe_prefix_params = [
        params
        for params in params_dict
        if params.startswith("fe_") or params == "outlet_column_name"
    ]

    columns_to_drop = []
    # Read off parameters with fe prefixes
    for fe_params in fe_prefix_params:
        logger.info(f"Reading {fe_params}")
        parameters_input = params_dict[fe_params]

        # Flatten multi dimension list into single list
        if isinstance(parameters_input, list):
            # List of list
            if all(isinstance(i, list) for i in parameters_input):
                parameters_input = list(itertools.chain(*parameters_input))
            # Simple list
            columns_to_drop.extend(parameters_input)

        elif isinstance(parameters_input, str):
            columns_to_drop.append(parameters_input)

        elif isinstance(parameters_input, Dict):
            columns_to_drop.extend(list(parameters_input.keys()))
        else:
            logger.info(
                f"The parameters {fe_params} processed is neither a str or list or Dict type, continuing to next parameter."
            )

    logger.info(
        f"The overall list of columns to drop are as follows: {columns_to_drop}"
    )
    if columns_to_drop:
        for partition_id, df in partitioned_data.items():
            validated_col_list = list(
                set(columns_to_drop).intersection(set(df.columns))
            )
            logger.info("Existing columns before dropping any")
            logger.info(
                f"Shape of data for {partition_id} before dropping {validated_col_list} is {df.shape}"
            )

            # Drop unwanted columns which are validated
            df.drop(columns=validated_col_list, inplace=True)

            # Drop rows with any nulls
            df.dropna(axis=0, how="any", inplace=True)

            logger.info(f"Shape of data for {partition_id} is {df.shape}")
            partitioned_data[partition_id] = df

    return partitioned_data


def reorder_data(
    partitioned_input_training: Dict[str, pd.DataFrame],
    partitioned_input_validation: Dict[str, pd.DataFrame],
    partitioned_input_testing: Dict[str, pd.DataFrame],
) -> OrdDict[str, pd.DataFrame]:
    """Function which pairs up files from training and validation subfolders containing fold-based merged outlets data in 'data' directory by using an OrderedDict to update corresponding alternating training, validation prefix folds before model training can be done.

    Args:
        partitioned_input_training (Dict[str, pd.DataFrame]): A dictionary with partition ids as keys and dataframe as values for training data.
        partitioned_input_validation (Dict[str, pd.DataFrame): A dictionary with partition ids as keys and dataframe as values for validation data.
        partitioned_input_testing (Dict[str, pd.DataFrame): A dictionary with partition ids as keys and dataframe as values for testing data.
    Returns:
        typing.OrderedDict[str, pd.DataFrame]): OrderedDict containing paired training and validation datasets updated in sorted partition ids.
    """
    ordered_dict = OrderedDict()

    for training_outlet_folds, validation_outlet_folds, testing_outlet_folds in zip(
        sorted(partitioned_input_training, reverse=False),
        sorted(partitioned_input_validation, reverse=False),
        sorted(partitioned_input_testing, reverse=False),
    ):
        logger.info(
            f"Pairing {training_outlet_folds}, {validation_outlet_folds}, {testing_outlet_folds} in order"
        )
        # Pair training and validation in such order as they are related by name. Only difference is the use of different prefixes, training and validation for their purpose.
        ordered_dict[training_outlet_folds] = partitioned_input_training[
            training_outlet_folds
        ]
        ordered_dict[validation_outlet_folds] = partitioned_input_validation[
            validation_outlet_folds
        ]
        ordered_dict[testing_outlet_folds] = partitioned_input_testing[
            testing_outlet_folds
        ]

    logger.info(f"Reordered files in the following order: {ordered_dict.keys()}.\n")
    return ordered_dict


def identify_const_column(
    partitioned_input: Dict[str, pd.DataFrame],
) -> Dict[str, List]:
    """Function which identifies the existence of constant column across dataframe residing in the partitioned_input dictionary.

    Args:
        partitioned_input (Dict[str, pd.DataFrame]): A dictionary with partition ids as keys and dataframe as values.

    Raises:
        None.

    Returns:
        Dict: Dictionary containing processed folds info (as key) and the corresponding column name information in a list (as values) which are constants.
    """

    # Instantiate an empty dictionary for storing constant column feature name
    const_column_dict = {}
    # Generates X,y datasets for each partition data based on mode set
    for (
        partition_id,
        partition_df,
    ) in partitioned_input.items():
        logger.info(f"Loading data from {partition_id}")

        # Load data from each partitioned file
        fold = partition_id.split("_")[1]

        const_column_list = [
            col for col in partition_df.columns if partition_df[col].nunique() == 1
        ]

        if const_column_list:
            logger.info(f"Found constant column(s): {const_column_list} for {fold}")
        else:
            logger.info(f"No constant column(s) found for {fold}")
        # Update dictionary
        const_column_dict[fold] = const_column_list

    logger.info("Completed checks on constant column(s) of dataframe.\n")
    return const_column_dict


def remove_const_column(
    partitioned_input: Dict[str, pd.DataFrame],
    const_column_dict: Dict[str, List],
) -> Dict[str, pd.DataFrame]:
    """Function which identifies columns with constant value (single unique value) and removes them from dataframes in partitioned_input containing fold-based merged outlets dataframe.

    Args:
        partitioned_input (Dict[str, pd.DataFrame]): A dictionary with partition ids as keys and dataframe as values.
        const_column_dict (Dict[str, List]): Dictionary containing fold-based columns which are constant (only 1 unique value)

    Returns:
        Dict: A dictionary with partition ids as keys and dataframe with constant columns removed as values.
    """

    # Instantiate an empty dictionary for storing X,y splits
    data_partition_dict = {}
    # Generates X,y datasets for each partition data based on mode set
    for (
        partition_id,
        partition_df,
    ) in partitioned_input.items():
        logger.info(f"Loading data from {partition_id}")

        # Load data from each partitioned file
        fold = partition_id.split("_")[1]
        logger.info(
            f"Before dropping any null: {partition_id} with shape {partition_df.shape}"
        )
        # Drop rows containing any nulls as entry as they are invalid.
        const_column_list = const_column_dict[fold]
        logger.info("Retrieved fold-based constant column information")

        # Drop learned constant columns based on training data in current dataset to ensure consistency. Only applies if such info is available
        if const_column_list:
            logger.info(f"Dropping {const_column_list}")
            partition_df.drop(columns=const_column_list, inplace=True)
        else:
            logger.info(f"No columns required to drop for this fold: {fold}")

        logger.info(f"Processed {partition_id} with shape {partition_df.shape}")
        # Update dict with processed dataframe
        data_partition_dict[partition_id] = partition_df
    logger.info("Completed necessary constant column(s) removal.\n")
    return data_partition_dict


def split_data_into_X_y_features(
    dataframe_dict: Dict[str, pd.DataFrame], params_dict: Dict[str, Any]
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    """Function which splits a provided dataframes residing in each entry of dataframe_dict based on a given target_column parameter into respective predictor and predicted components.

    Depending on the prefix of the input dataframe_dict keys, the dataframes are then categorised into training/validation/testing categories. It assumes the target_column passed in is valid.

    Args:
        partitioned_input (Dict[str, pd.DataFrame]): A dictionary with partition ids as keys and dataframe as values.
        params_dict (Dict[str, Any]): Dictionary referencing parameters.yml.

    Raises:
        None.

    Returns:
        Tuple[Dict[str, pd.DataFrame],Dict[str, pd.DataFrame]]: Tuple of dictionaries similar to partitioned_input representing training and validation datasets
    """
    target_column = params_dict["target_column_for_modeling"]

    # Dictionary to store train/val/test partitions
    train_data_partition_dict = {}
    val_data_partition_dict = {}
    test_data_partition_dict = {}

    for df_id, df in dataframe_dict.items():
        if target_column not in df.columns:
            logger.error(
                f"The specified target column for modeling {target_columns} is not found. Returning empty dictionary.\n"
            )
            return data_partition_dict
        logger.info(
            "Separating features and target columns. Object based features will be dropped. Also columns with null will be dropped to avoid model training issues."
        )

        y_df = df[[target_column]]
        X_df = df.drop(columns=target_column).select_dtypes(exclude=["object"])

        # Drop columns with null. This could be caused by unseen encoding which may not appear in training data but instead validation data due to time period differences.
        nan_cols_list = [i for i in X_df.columns if X_df[i].isnull().any()]
        if nan_cols_list:
            logger.info(
                f"Columns with nulls identified: {nan_cols_list} after selecting non-object datatype."
            )
        X_df.drop(columns=nan_cols_list, inplace=True)

        # Convert filename structure training_fold1_expanding_window_param_0_X
        # to training_fold1_expanding_window
        filename_substring = "_".join(df_id.split("_")[0:5])
        # Update dict with splits

        logger.info(
            f"Assigning {df_id} to either training/validation/testing partitions..."
        )
        if df_id.startswith("training"):
            train_data_partition_dict[f"{filename_substring}_X"] = X_df
            train_data_partition_dict[f"{filename_substring}_y"] = y_df
        elif df_id.startswith("validation"):
            val_data_partition_dict[f"{filename_substring}_X"] = X_df
            val_data_partition_dict[f"{filename_substring}_y"] = y_df
        elif df_id.startswith("testing"):
            test_data_partition_dict[f"{filename_substring}_X"] = X_df
            test_data_partition_dict[f"{filename_substring}_y"] = y_df
        logger.info(
            f"Created features of shape {X_df.shape} and target {y_df.shape} for {df_id}"
        )

    logger.info("Completed feature/target splits for each fold.\n")
    return train_data_partition_dict, val_data_partition_dict, test_data_partition_dict
