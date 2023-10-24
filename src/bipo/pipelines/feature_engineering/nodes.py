import pandas as pd
import logging
from typing import List, Dict, Any, Union
from datetime import datetime
from bipo import settings
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder

## Encodings
from .encoding import (
    ordinal_encoding_fit,
    ordinal_encoding_transform,
    one_hot_encoding_fit,
    one_hot_encoding_transform,
)

## Lag generations
from .lag_feature_generation import (
    create_simple_lags,
    create_sma_lags,
    create_lag_weekly_avg_sales,
)

from .binning import equal_freq_binning_fit, equal_freq_binning_transform

## Normalization
from .standardize_normalize import standard_norm_fit, standard_norm_transform

logger = logging.getLogger("kedro")


def generate_lag(
    data_preprocessed_outlet_partitions: Dict[str, pd.DataFrame],
    params_dict: Dict[str, Any],
) -> Dict[str, pd.DataFrame]:
    """Creates a dataframe containing generated lag values based on column_name variable representing dataframe column name according to params_dict containing lag related parameters.

    Args:
        data_preprocessed_outlet_partitions (Dict[str, pd.DataFrame]): A dictionary with partition ids as keys and dataframe as values.
        params_dict (Dict[str, Any]): Dictionary of parameters as referenced from parameters.yml.

    Raises:
        None.

    Returns:
        Dict[str, pd.DataFrame]: Dictionary containing generated dataframe with only lag columns, prefix by 'id' as an identifier.
    """
    col_to_create_lag_set = set(params_dict["columns_to_create_lag_features"])
    lag_df_partition_dict = {}

    if not col_to_create_lag_set:
        logger.info(
            "No columns specified for lag feature creation. Skipping lag creation process.\n"
        )

        return lag_df_partition_dict

    logger.info(f"Column specified for lag generation {col_to_create_lag_set}...")

    # Parameters for lag/SMA/lag week periods
    lag_periods_list = params_dict["lag_periods_list"]
    sma_window_period_list = params_dict["sma_window_periods_list"]
    sma_shift_period = params_dict["sma_tsfresh_shift_period"]
    lag_week_periods_list = params_dict["lag_week_periods_list"]

    # Create partition_dict for gnerated lag df for each outlet
    for (
        outlet_partition_id,
        outlet_partition_load_df,
    ) in data_preprocessed_outlet_partitions.items():
        df = outlet_partition_load_df
        df.index = pd.to_datetime(df.index, format="%Y-%m-%d")
        df_col_set = set(list(df.columns))
        logger.info(f"Dataframe columns: {df_col_set}")

        # Find intersection of both sets to determine which features are available to support lag generations
        valid_lag_col_list = list(col_to_create_lag_set.intersection(df_col_set))

        # If no columns is identified, stop and proceed to next
        if not valid_lag_col_list:
            logger.info(
                f"No valid columns to generate lags for {outlet_partition_id}.\n"
            )
            return lag_df_partition_dict

        logger.info(f"Identified {valid_lag_col_list} columns to generate lag")
        # Create a df subset copy comprising of valid specified columns to create lags
        lag_df = df[valid_lag_col_list].copy()

        logger.info(f"Dataframe shape {lag_df.shape} which lags is to be generated")
        # Function call to various lag features generated

        # Simple lag
        simple_lag_df = create_simple_lags(df=lag_df, lag_periods_list=lag_periods_list)

        logger.info(f"Generated simple lag dataframe: {simple_lag_df.shape}")
        # Sma lag
        sma_lag_df = create_sma_lags(
            df=lag_df,
            shift_period=sma_shift_period,
            sma_window_period_list=sma_window_period_list,
        )
        logger.info(
            f"Generated simple moving average lag dataframe: {sma_lag_df.shape}"
        )

        # Simple lag
        simple_lag_df = create_simple_lags(df=lag_df, lag_periods_list=lag_periods_list)

        logger.info(f"Generated simple lag dataframe: {simple_lag_df.shape}")
        # Sma lag
        sma_lag_df = create_sma_lags(
            df=lag_df,
            shift_period=sma_shift_period,
            sma_window_period_list=sma_window_period_list,
        )
        logger.info(
            f"Generated simple moving average lag dataframe: {sma_lag_df.shape}"
        )

        # Lag average weekly df
        lag_weekly_avg_df = create_lag_weekly_avg_sales(
            df=lag_df,
            shift_period=sma_shift_period,
            num_weeks_list=lag_week_periods_list,
        )
        logger.info(
            f"Generated lag weekly average dataframe: {lag_weekly_avg_df.shape}"
        )
        # Join all generated lag features on index and drop rows with nans
        lag_generated_df = pd.concat(
            [simple_lag_df, sma_lag_df, lag_weekly_avg_df], axis=1, ignore_index=False
        )

        lag_generated_df.dropna(axis=0, inplace=True)
        # Create new partition_id for reference in merge_fold_and_generated_lag function.
        outlet_info = outlet_partition_id.split("_")[-1]
        new_partition_id = f"lag_{outlet_info}"
        lag_df_partition_dict[new_partition_id] = lag_generated_df

    return lag_df_partition_dict


def merge_fold_and_generated_lag(
    fold_based_outlet_partitioned_data: Dict[str, pd.DataFrame],
    generated_lags_dict: Dict[str, pd.DataFrame],
) -> Dict[str, pd.DataFrame]:
    """Function which merges a dictionary containing calculated outlet lagged values into a dictionary containing fold-based outlet partitioned dataframe via left merge (left dataframe referring to fold-based outlet partitioned dataframe).

    Args:
        fold_based_outlet_partitioned_data (Dict[str, pd.DataFrame): A dictionary with partition ids as keys and dataframe as values.
        generated_lags_dict (Dict[str, pd.DataFrame]): A dictionary with partition ids as keys and dataframe as values.

    Raise:
        None.

    Returns:
        Dict[str, pd.DataFrame]: Dictionary containing updated dataframe with lagged features if applicable. Else, return fold_based_outlet_partitioned_data.
    """
    if not generated_lags_dict:
        return fold_based_outlet_partitioned_data

    # Loop through the dictionary and join the outlet dataframe with lag values with the fold-based outlet dataframes to create an updated dataframe with lagged features.
    for (
        fold_based_outlet_id,
        fold_based_outlet_df,
    ) in fold_based_outlet_partitioned_data.items():
        # Fold based id is of the form eg. training_fold1_expanding_window_param_90_305
        outlet_info = fold_based_outlet_id.split("_")[-1]

        # Get the corresponding outlet dataframe.
        generated_lag_df = generated_lags_dict[f"lag_{outlet_info}"]

        fold_based_outlet_df.index = pd.to_datetime(
            fold_based_outlet_df.index, format="%Y-%m-%d"
        )
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


def apply_feature_encoding_transform(
    partitions_dict: Dict[str, pd.DataFrame],
    params_dict: Dict[str, Any],
    feature_encoding_dict: Dict[str, Union[OrdinalEncoder, OneHotEncoder]],
) -> pd.DataFrame:
    """This function applies necessary encoding transform which includes both one-hot encoding and categorical encoding. Can be extended to include encoding functions if defined under apply_feature_encoding_fit function.

    Args:
        partitions_dict (Dict[str, pd.DataFrame]): A dictionary with partition ids as keys and dataframe as values.
        params_dict (Dict[str, Any]): Dictionary referencing parameters.yml.
        feature_encoding_dict (Union[OrdinalEncoder, OneHotEncoder]): Dictionary containing new/updated encoding parameters which will be used for applying necessary transforms.

    Raises:
        None.

    Returns:
        pd.DataFrame: Dataframe with encoded features.

    """
    # Read parameters.yml
    logger.info(
        "Retrieving provided onehotencoding/ordinal features and value information"
    )

    ordinal_encoding_dict = params_dict["fe_ordinal_encoding_dict"]
    one_hot_encoding_col_list = params_dict["fe_one_hot_encoding_col_list"]

    for partition_id, partition_fold_df in partitions_dict.items():
        # Ordinal encoding section
        fold_info = partition_id.split("_")[1]
        if not ordinal_encoding_dict:
            logger.info(
                "No columns specified for ordinal encoding. Skipping ordinal encoding for current partition."
            )
            df = partition_fold_df
        else:
            # Extract fold and outlet information when doing encoding based on filename as represented by partition_id
            df = ordinal_encoding_transform(
                df=partition_fold_df,
                ordinal_encoder=feature_encoding_dict[f"{fold_info}_ord"],
            )

        if not one_hot_encoding_col_list:
            logger.info("No columns specified for one-hot encoding.Skipping process.")
        else:
            # Extract fold and outlet information when doing encoding based on filename as represented by partition_id
            logger.info(f"Applied one-hot-encoding transform for each {fold_info}...")
            df = one_hot_encoding_transform(
                df=df,
                ohe_encoder=feature_encoding_dict[f"{fold_info}_ohe"],
            )

        partitions_dict[partition_id] = df
    logger.info("Completed encoding transforms.\n")
    return partitions_dict


def apply_feature_encoding_fit(
    partitions_dict: Dict[str, pd.DataFrame],
    params_dict: Dict[str, Any],
) -> Dict[str, Union[OrdinalEncoder, OneHotEncoder]]:
    """This function applies necessary encoding which includes one-hot encoding and categorical encoding. Additional encodings can be added into this function if required.

    Args:
        partitions_dict (Dict[str, pd.DataFrame]): A dictionary with partition ids as keys and dataframe as values.
        params_dict (Dict[str, Any]): Dictionary referencing parameters.yml.

    Raises:
        None.

    Returns:
        Dict[str, Any]: Dictionary containing new/updated encoding parameters encompassing sklearn OrdinalEncoder and OneHotEncoder. Other encodings can be included if they are implemented in this function.

    """
    # Read parameters.yml
    logger.info(
        "Retrieving provided onehotencoding/ordinal features and value information..."
    )
    ordinal_col_dict = params_dict["fe_ordinal_encoding_dict"]
    one_hot_encoding_col_list = params_dict["fe_one_hot_encoding_col_list"]

    # Instantiate empty dict to store encoding parameters involving one-hot-encoding and ordinal-encoding
    feature_encoding_dict = {}

    # Ordinal encoding section
    for partition_id, partition_fold_df in partitions_dict.items():
        if not ordinal_col_dict:
            logger.info("No columns specified for ordinal encoding")
        else:
            fold_info = partition_id.split("_")[1]
            # Extract fold and outlet information when doing encoding based on filename as represented by partition_id
            feature_encoding_dict = ordinal_encoding_fit(
                df=partition_fold_df,
                ordinal_columns_dict=ordinal_col_dict,
                ordinal_encoding_dict=feature_encoding_dict,
                fold=fold_info,
            )
        logger.info(
            f"Ordinal encoding parameters learned and encoding dictionary is updated for {partition_id}"
        )

        if not one_hot_encoding_col_list:
            logger.info("No columns specified for one-hot encoding.")
        else:
            # Extract fold and outlet information when doing encoding based on filename as represented by partition_id
            fold_info = partition_id.split("_")[1]
            outlet_info = partition_id.split("_")[-1]
            logger.info("Learning one-hot-encoding for each fold and outlet...")
            feature_encoding_dict = one_hot_encoding_fit(
                df=partition_fold_df,
                ohe_column_list=one_hot_encoding_col_list,
                ohe_encoding_dict=feature_encoding_dict,
                fold=fold_info,
            )
    logger.info("Necessary encoding fitting completed.\n")

    return feature_encoding_dict


def apply_standard_norm_fit(
    partitions_dict: Dict[str, pd.DataFrame],
    params_dict: Dict[str, Any],
) -> Dict[str, Any]:
    """This function applies required standardise/normalisation fitting as configured in parameters.yml on dataframe partitions representing outlet datasets and saves them into a dictionary containing learned parameters.

    Args:
        partitions_dict (Dict[str, pd.DataFrame]): A dictionary with partition ids as keys and dataframe as values.
        params_dict (Dict[str, Any]): Dictionary referencing parameters.yml.

    Raises:
        None.

    Returns:
        Dict[str, Any]: Dictionary containing standardisation/normalisation parameters.

    """
    # Read parameters.yml
    logger.info("Retrieving normalization approach...")
    std_norm_approach = params_dict["normalization_approach"]
    columns_to_norm_list = params_dict["columns_to_std_norm_list"]

    # If accidental non-list input is used in parameters.yml
    if not isinstance(columns_to_norm_list, List):
        columns_to_norm_list = []

    # Instantiate a empty dict to store encoding parameters
    std_norm_encoding_dict = {}

    # Get bool state of whether to include lag features
    include_lag_state = params_dict["include_lags_columns_for_std_norm"]

    # Extend columns_to_norm_list if lag features need to be included. As the columns for all dataframe in the data partitions are the same, using first instance would suffice.
    if include_lag_state:
        first_partition_id = next(iter(partitions_dict))
        first_partition_df = partitions_dict[first_partition_id]
        # Filter out lag prefixed columns which are floats/int by default.
        lag_col_list = [
            col for col in first_partition_df.columns if col.startswith("lag_")
        ]
        columns_to_norm_list.extend(lag_col_list)

    if not columns_to_norm_list:
        logger.info(
            "No columns to apply standard/normalisation transformation. Skipping this process.\n"
        )
    else:
        # Ordinal encoding section. Loop update for std_norm_encoding_dict for each outlet-fold
        for partition_id, partition_df in partitions_dict.items():
            fold_info = partition_id.split("_")[1]
            outlet_info = partition_id.split("_")[-1]
            logger.info(
                f"Fitting selected normalisation approach for fold: {fold_info}, outlet: {outlet_info}"
            )

            std_norm_encoding_dict = standard_norm_fit(
                df=partition_df,
                col_to_std_or_norm_list=columns_to_norm_list,
                std_norm_encoding_dict=std_norm_encoding_dict,
                option=std_norm_approach,
                fold=fold_info,
                outlet=outlet_info,
            )

        logger.info(f"Standardised/Normalisation fitting of data completed.\n")
    return std_norm_encoding_dict


def apply_standard_norm_transform(
    partitions_dict: Dict[str, pd.DataFrame],
    params_dict: Dict[str, Any],
    std_norm_encoding_dict: Dict[str, Union[OrdinalEncoder, OneHotEncoder]],
) -> Dict[str, pd.DataFrame]:
    """This function applies preferred standardise/normalisation as configured in parameters.yml.

    Args:
        partitions_dict (Dict[str, pd.DataFrame]): A dictionary with partition ids as keys and dataframe as values.
        params_dict (Dict[str, Any]): Dictionary referencing parameters.yml.
        std_norm_dict (Dict[str, Any]): Dictionary containing new/updated sklearn StandardScaler/Normalizer parameters.

    Raises:
        None.

    Returns:
        Dict[str, pd.DataFrame]: Dictionary containing new/updated dataframe with unchanged key as per input.

    """
    # Read parameters.yml
    logger.info("Retrieving normalization approach...")
    std_norm_approach = params_dict["normalization_approach"]
    columns_to_norm_list = params_dict["columns_to_std_norm_list"]

    # If accidental non-list input is used in parameters.yml
    if not isinstance(columns_to_norm_list, List):
        columns_to_norm_list = []

    # Get bool state of whether to include lag features
    include_lag_state = params_dict["include_lags_columns_for_std_norm"]

    # Extend columns_to_norm_list if lag features need to be included. As the columns for all dataframe in the data partitions are the same, using first instance would suffice.
    if include_lag_state:
        first_partition_id = next(iter(partitions_dict))
        first_partition_df = partitions_dict[first_partition_id]
        # Filter out lag prefixed columns which are floats/int by default.
        lag_col_list = [
            col for col in first_partition_df.columns if col.startswith("lag_")
        ]
        columns_to_norm_list.extend(lag_col_list)

    if not columns_to_norm_list:
        logger.info(
            "No columns to apply standard/normalisation transformation. Skipping this process.\n"
        )
    else:
        # Apply standardisation/normalisation transformation using dictionary referencing to extract learned sklearn standardscaler/normalizer object
        for partition_id, partition_df in partitions_dict.items():
            fold_info = partition_id.split("_")[1]
            outlet_info = partition_id.split("_")[-1]
            logger.info(
                f"Applying {std_norm_approach} transform for {fold_info} and outlet {outlet_info}"
            )

            std_norm_df = standard_norm_transform(
                df=partition_df,
                std_norm_object=std_norm_encoding_dict[
                    f"{fold_info}_{outlet_info}_{std_norm_approach}"
                ],
            )
            partitions_dict[partition_id] = std_norm_df

        logger.info(f"Completed {std_norm_approach} transform for all outlets.\n")
    return partitions_dict


def apply_binning_fit(
    partitions_dict: Dict[str, pd.DataFrame],
    params_dict: Dict[str, Any],
) -> Dict[str, List]:
    """This function applies binning techniques, specifically equal-frequency binning, to designated columns in the provided data partitions, updating a dictionary with learned binning parameters.

    Args:
        partitions_dict (Dict[str, pd.DataFrame]): A dictionary with partition ids as keys and dataframe as values.
        params_dict (Dict[str, Any]): Dictionary referencing parameters.yml.

    Raises:
        None

    Returns:
        Dict[str, List]: Dictionary containing new/updated encoding parameters.

    """
    # Read parameters.yml
    logger.info(
        "Retrieving provided onehotencoding/ordinal features and value information..."
    )
    binning_col_dict = params_dict["binning_dict"]

    # Instantiate an empty dict to store binning edges parameters learned
    binning_encoding_dict = {}

    if not binning_col_dict:
        logger.info("No columns specified for binning.Skipping process.\n")
    else:
        # Ordinal encoding section
        for partition_id, partition_df in partitions_dict.items():
            fold_info = partition_id.split("_")[1]
            outlet_info = partition_id.split("_")[-1]
            logger.info(f"Applying binning to {partition_id}")
            # Apply binning for each column
            for bin_col, bin_labels in binning_col_dict.items():
                logger.info(
                    f"Extracting binning parameters through equal frequency binning for {bin_col}"
                )
                partition_df.dropna(subset=[bin_col], axis=0, inplace=True)
                binning_encoding_dict = equal_freq_binning_fit(
                    df=partition_df,
                    bin_column_name=bin_col,
                    bin_labels_list=bin_labels,
                    fold=fold_info,
                    outlet=outlet_info,
                    binning_encoding_dict=binning_encoding_dict,
                )
        logger.info("Updated dictionary with learned binning parameters.")

    return binning_encoding_dict


def apply_binning_transform(
    partitions_dict: Dict[str, pd.DataFrame],
    params_dict: Dict[str, Any],
    binning_encoding_dict: Dict[str, Any],
) -> Dict[str, pd.DataFrame]:
    """This function executes the binning categorisation using a provided binning_encoding_dict containing binning related parameters, which is generated under apply_binning_fit function above.

    Args:
        partitions_dict (Dict[str, pd.DataFrame]): Dictionary containing file represented by fold based partitions of a selected data split source as defined in Kedro framework.
        params_dict (Dict[str, Any]): Dictionary referencing key-values from parameters.yml.
        binning_encoding_dict (Dict[str, Any]): Dictionary containing new/updated encoding parameters.

    Raises:
        None.

    Returns:
        Dict[str, pd.DataFrame]: Dictionary containing new/updated dataframe with unchanged key as per input.

    """
    # Read parameters.yml
    logger.info(
        "Retrieving provided onehotencoding/ordinal features and value information..."
    )
    binning_col_dict = params_dict["binning_dict"]
    if not binning_encoding_dict or not binning_col_dict:
        logger.info(
            "No learned binning parameters available. Skipping binning transformation process.\n"
        )
    else:
        # Ordinal encoding section
        for partition_id, partition_df in partitions_dict.items():
            fold_info = partition_id.split("_")[1]
            outlet_info = partition_id.split("_")[-1]
            for bin_col, bin_labels in binning_col_dict.items():
                logger.info(
                    f"Extracting binning parameters through equal frequency binning for {fold_info} and outlet {outlet_info}"
                )
                partition_df = equal_freq_binning_transform(
                    df=partition_df,
                    bin_column_name=bin_col,
                    bin_labels_list=bin_labels,
                    bin_edges_list=binning_encoding_dict[f"{fold_info}_{outlet_info}"],
                )
                # Update dictionary with updated dataframe
                partitions_dict[partition_id] = partition_df
    logger.info("Completed binning transform.\n")
    return partitions_dict


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
