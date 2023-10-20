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
from kedro.io import DataSetError
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
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
        partitioned_data (Dict[str, pd.DataFrame]): Kedro MemoryDataSet dictionary containing individual outlet related features dataframe as values with filename using outlet as identifier.
        params_dict (Dict): Dictionary referencing parameters.yml.


    Raises:
        None

    Returns:
        Dict[str, pd.DataFrame]: Dictionary containing processed dataframe identified by its input identifier.
    """

    # Identify parameters with 'fe' prefixes and outlet_column_name key
    fe_prefix_params = [
        params
        for params in params_dict
        if params.startswith("fe") or params == "outlet_column_name"
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
            logger.info(
                f"Shape of data for {partition_id} before dropping {validated_col_list} is {df.shape}"
            )

            # Drop unwanted columns which are validated
            df.drop(columns=validated_col_list, inplace=True)

            # Drop rows with nulls
            df.dropna(axis=0, how="all", inplace=True)

            logger.info(f"Shape of data for {partition_id} is {df.shape}")
            partitioned_data[partition_id] = df

    return partitioned_data


def concat_same_folds_of_outlet_data(
    partitioned_input: Dict[str, pd.DataFrame]
) -> Dict[str, pd.DataFrame]:
    """Function that loads and concatenates vertically all outlet data belonging to the same fold together.

    Args:
        partitioned_input (Dict[str, pd.DataFrame]): Kedro IncrementalDataSet dictionary containing individual outlet related features dataframe as values with filename using outlet as identifier.

    Raises:
        None

    Returns:
        Dict[str, pd.DataFrame]: Dictionary containing concatenated fold-based outlet data.
    """
    # Dummy variable to track fold number being processed so as to ensure concatenation is applied to folds with same number.
    current_fold = False
    merged_fold_outlet_dict = {}

    for partition_id, partition_df in sorted(partitioned_input.items(), reverse=False):
        # Extract fold_information from the partition_id. Create new empty dataframe when new fold is processed with a new shortened partition id. This shortened partition converts fold partition files e.g 'testing_fold1_expanding_window_param_90' to 'testing_fold1_expanding_window_param'.
        new_fold = partition_id.split("_")[1]
        if new_fold != current_fold:
            merged_outlet_df = pd.DataFrame()
            logger.info(f"Processing fold: {new_fold}")

            # Construct new identifier with outlet id removed.
            new_partition_id = "_".join(partition_id.split("_")[:-2])

        # Merge loaded

        partition_data = partition_df
        merged_outlet_df = pd.concat([merged_outlet_df, partition_data], join="outer")
        logger.info(f"Concatenated {partition_id} to existing dataframe.")
        current_fold = new_fold
        merged_fold_outlet_dict[new_partition_id] = merged_outlet_df

    logger.info(f"Concatenated all outlets of each fold.")
    return merged_fold_outlet_dict


def reorder_data(
    partitioned_input_training: Dict[str, pd.DataFrame],
    partitioned_input_validation: Dict[str, pd.DataFrame],
) -> OrdDict[str, pd.DataFrame]:
    """Function which pairs file from training and validation subfolders by using an OrderedDict to update corresponding alternating training, validation prefix folds before model training can be done.

    Args:
        partitioned_input_training (Dict[str, pd.DataFrame]): Kedro MemoryDataset Dictionary containing training folds of data.
        partitioned_input_validation (Dict[str, pd.DataFrame): Kedro MemoryDataset Dictionary containing validation folds of data

    Returns:
        typing.OrderedDict[str, pd.DataFrame]): Dictionary containing paired ordering of training and validation outlets datasets.
    """
    ordered_dict = OrderedDict()

    for training_outlet_folds, validation_outlet_folds in zip(
        sorted(partitioned_input_training, reverse=False),
        sorted(partitioned_input_validation, reverse=False),
    ):
        logger.info(
            f"Pairing {training_outlet_folds}, {validation_outlet_folds} in order"
        )
        # Pair training and validation in such order as they are related by name. Only difference is the use of different prefixes, training and validation for their purpose.
        ordered_dict[training_outlet_folds] = partitioned_input_training[
            training_outlet_folds
        ]
        ordered_dict[validation_outlet_folds] = partitioned_input_validation[
            validation_outlet_folds
        ]

    logger.info(f"Reordered files in the following order: {ordered_dict.keys()}.\n")
    return ordered_dict


def identify_const_column(
    partitioned_input: Dict[str, pd.DataFrame],
    params_dict: Dict[str, Any],
) -> Dict[str, List]:
    """Function which applies necessary model specific preprocessing steps based on the model parameters defined in params_dict and subsequently splits data into X (feature), y (target) components.

    Args:
        partitioned_input (Dict[str, pd.DataFrame]): Non-PartitionedDataset
        dictionary containing file, and dataframe.
        params_dict (Dict): Dictionary referencing parameters.yml.

    Returns:
        Dict: Dictionary containing processed folds, containing column information which are constants.
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
        # Handle model specific preprocessing:
        logger.info(f"Dropping rows with nulls for {partition_id}")
        partition_df.dropna(how="any", inplace=True)

        const_column_list = list(partition_df.columns[partition_df.nunique() == 1])
        logger.info(f"Constant columns: {const_column_list}")

        # Update dictionary
        const_column_dict[fold] = const_column_list

    logger.info("Completed model specific data preprocessing.\n")
    return const_column_dict


def remove_const_column(
    partitioned_input: Dict[str, pd.DataFrame],
    params_dict: Dict[str, Any],
    const_column_dict: Dict[str, List],
) -> Dict[str, pd.DataFrame]:
    """Function which applies necessary model specific preprocessing steps based on the model parameters defined in params_dict and subsequently splits data into X (feature), y (target) components.

    Args:
        partitioned_input (Dict[str, pd.DataFrame]): Non-PartitionedDataset dictionary containing file, and dataframe.
        params_dict (Dict): Dictionary referencing parameters.yml.
        const_column_dict (Dict[str, List]): Dictionary containing fold-based columns which are constant (only 1 unique value)

    Returns:
        Dict: Dictionary containing processed dataframe folds.
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

        # Drop learned constant columns based on training data in current dataset to ensure consistency.
        if const_column_list:
            logger.info(f"Dropping {const_column_list}")
            partition_df.drop(columns=const_column_list, inplace=True)
        else:
            logger.info(f"No columns required to drop for this fold: {fold}")

        logger.info(f"Processed {partition_id} with shape {partition_df.shape}")
        # Update dict with processed dataframe
        data_partition_dict[partition_id] = partition_df
    logger.info("Completed model specific data preprocessing.\n")
    return data_partition_dict


def split_data_into_X_y_features(
    dataframe_dict: Dict[str, str], params_dict: Dict[str, Any]
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    """Function which splits a provided dataframe residing in each entry of dataframe_dict by a given target_column parameters into feature and target components.

    Depending on the prefix of the input dataframe_dict keys, the dataframes are then categorised into training/validation/testing categories. It assumes the target_column passed in is valid.

    Args:
        partitioned_input (Dict[str, str]): Dictionary containing file,         lazy-loading function key-value mapping based on Kedro framework.
        params_dict (Dict[str, Any]): Dictionary referenced from parameters.yml

    Returns:
        Tuple[Dict[str, pd.DataFrame],Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]: Tuple of dictionary representing training, validation and testing datasets
    """
    target_column = params_dict["target_column_for_modeling"]
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
        X_df = df.drop(target_column, axis=1).select_dtypes(exclude=["object"])

        # Drop columns with null. This could be caused by unseen encoding which may not appear in training data but instead validation data due to time period differences. 
        nan_cols_list = [i for i in X_df.columns if X_df[i].isnull().any()]
        logger.info(f"Identified columns with nulls {nan_cols_list} after selecting non-object datatype.")
        X_df.drop(columns=nan_cols_list, inplace=True)

        # Convert filename structure training_fold1_expanding_window_param_0_X
        # to training_fold1_expanding_window
        filename_substring = "_".join(df_id.split("_")[0:5])
        # Update dict with splits
        if df_id.startswith("train"):
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


def merge_tsfresh_features_with_outlets(
    partitioned_outlet_input: Dict[str, pd.DataFrame],
    partitioned_tsfresh_input: Dict[str, pd.DataFrame],
) -> Dict[str, pd.DataFrame]:
    """Function which left merges a outlet-based generated tsfresh features to each outlet based on common time-index. As the id used for both partitioned inputs are the same, joining can be simply made by using one of the id retreived from either of the inputs and doing a direct referencing from both dictionary keys.

    Args:
        partitioned_outlet_input: Dict[str, pd.DataFrame]: Dataframe partitioned by outlets loaded in Kedro MemoryDataSet on the fly without any physical file presence or Kedro PartitionedDataSet as a source.
        partitioned_mmm_input: Dict[str, pd.DataFrame]: PartitionedDataset of fold-based tsfresh features dataframe.

    Raises:
        None.

    Returns:
        Union[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]: Dictionary containing merged dataframes involving both tsfresh features and input outlet dataframe features. Otherwise, returns partitioned_outlet_input with dataframe lazy loading function.
    """
    if not partitioned_tsfresh_input:
        logger.info(
            "No tsfresh features to merge, skipping merging of such features to outlet dataframes.\n"
        )
        return partitioned_outlet_input
    logger.info("Merging tsfresh features with fold outlets...")
    merged_partitioned_dict = {}
    for partitioned_id, partitioned_df in partitioned_outlet_input.items():
        outlet_fold_df = partitioned_df

        outlet_fold_df.index = pd.to_datetime(outlet_fold_df.index, format="%Y-%m-%d")

        # Load the corresponding fold in lightweightmmm partitions
        tsfresh_fold_df = partitioned_tsfresh_input[partitioned_id]()
        tsfresh_fold_df.index = pd.to_datetime(tsfresh_fold_df.index, format="%Y-%m-%d")

        outlet_fold_tsfresh_df = outlet_fold_df.merge(
            tsfresh_fold_df,
            how="left",
            left_index=True,
            right_index=True,
            suffixes=("", "_y"),
        )

        # Drop excess columns with _y suffix due to emrge
        col_with_y_suffix = [
            col for col in outlet_fold_tsfresh_df if col.endswith("_y")
        ]
        outlet_fold_tsfresh_df.drop(columns=col_with_y_suffix, inplace=True)

        merged_partitioned_dict[partitioned_id] = outlet_fold_tsfresh_df
    logger.info("Completed Tsfresh features merged with folds.\n")
    return merged_partitioned_dict


def merge_mmm_features_with_fold_outlets(
    partitioned_outlet_input: Dict[str, pd.DataFrame],
    partitioned_mmm_input: Dict[str, pd.DataFrame],
) -> Dict[str, pd.DataFrame]:
    """Function which left merges a fold-based generated LightweightMMM features to each outlet based on common time-index. This is on the assumption that LightweightMMM features are applicable across all-outlets regardless of outlet types.

    Args:
        partitioned_outlet_input (Dict[str, pd.DataFrame]): Dataframe partitioned by outlets from Kedro PartitionedDataSet.
        partitioned_mmm_input (Dict[str, pd.DataFrame]): Generated fold-based LightweightMMM features dataframe partitioned by folds.

    Raises:
        None.

    Returns:
        Dict[str, pd.DataFrame]: Dictionary containing merged dataframes involving both lightweightMMM features and input outlet dataframe features. Otherwise, returns partitioned_outlet_input with dataframe lazy loading function.

    """
    if not partitioned_mmm_input:
        logger.info(
            "No lightweightMMM features to merge, skipping merging of such features to outlet dataframes.\n"
        )
        return partitioned_outlet_input
    merged_partitioned_dict = {}
    # Catch cases when partitioned_mmm_input is empty
    logger.info("Merging lightweightmmm features with fold outlets...")
    for partitioned_id, partitioned_df in partitioned_outlet_input.items():
        # Retrieve fold information from partitioned_outlet_input
        fold_info = partitioned_id.split("_")[1]

        outlet_fold_df = partitioned_df

        outlet_fold_df.index = pd.to_datetime(outlet_fold_df.index, format="%Y-%m-%d")

        # Load the corresponding fold in lightweightmmm partitions
        mmm_fold_df = partitioned_mmm_input[fold_info]
        mmm_fold_df.index = pd.to_datetime(mmm_fold_df.index, format="%Y-%m-%d")

        outlet_fold_mmm_df = outlet_fold_df.merge(
            mmm_fold_df,
            how="left",
            left_index=True,
            right_index=True,
            suffixes=("", "_y"),
        )

        # Drop excess columns with _y suffix due to emrge
        col_with_y_suffix = [col for col in outlet_fold_mmm_df if col.endswith("_y")]
        outlet_fold_mmm_df.drop(columns=col_with_y_suffix, inplace=True)
        merged_partitioned_dict[partitioned_id] = outlet_fold_mmm_df
    logger.info("Completed LightweightMMM features merged with folds.\n")
    return merged_partitioned_dict


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
