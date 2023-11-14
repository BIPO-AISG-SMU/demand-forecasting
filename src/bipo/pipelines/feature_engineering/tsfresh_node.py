import pandas as pd
import re
from typing import Dict, Union, Any, List
from kedro.config import ConfigLoader
import tsfresh
from .tsfresh_main import TsfreshFe
from bipo import settings
import logging

logger = logging.getLogger(settings.LOGGER_NAME)

# Instantiate config
conf_loader = ConfigLoader(conf_source=settings.CONF_SOURCE)
constants = conf_loader.get("constants*")


# 1st node
# feature selection pipeline on single outlet dataframe
def run_tsfresh_feature_selection_process(
    partitioned_input: Dict[str, pd.DataFrame], params_dict: Dict[str, Any]
) -> Dict[str, Dict]:
    """Function which executes tsfresh feature selection to select relevant tsfresh features on the dataset as defined by partitioned_input. These relevant features are then stored in a dictionary by fold and outlet.

    Args:
        partitioned_input (Dict[str,pd.DataFrame]): partitioned input for training fold outlet dataframe.
        params_dict (Dict[str, Any]): Parameters referencing parameters.yml.

    Returns:
        Dict[str, Dict]: Dictionary containing fold-output identifier as key name and a dictionary of relevant tsfresh features as values as values. i.e dictionary of dictionary.
    """
    # Read yml
    logger.info("Applying tsfresh feature selection")

    target_feature = params_dict[
        "fe_target_feature_name"
    ]  # Raw target column based on original data which is not processed (i.e binned, etc.)

    # run on training folds, and output relevant_feature artefact for each fold
    # filter outlet by fold
    fold_list = [filename.split("_")[1] for filename in partitioned_input.keys()]
    fold_list = list(set(fold_list))

    # create dictionary to store json
    artefact_dict = {}
    # iterate over training fold outlet dataframes to prevent data leakage
    for fold in fold_list:
        # filter outlets by fold
        outlet_fold_dict = {
            key: partitioned_input[key]
            for key in partitioned_input.keys()
            if fold in key
        }
        # create list to store relevance tables for each fold
        relevance_table_list = []
        logger.info(f"Starting tsfresh feature selection on {fold}....")

        # iterate over outlets in specific fold and generate relevance table
        for outlet in outlet_fold_dict:
            logger.info(f"Feature selection on {fold}, outlet {outlet}")
            outlet_df = outlet_fold_dict[outlet]
            # Get relevant features for tsfresh based on each outlets in a specific fold, referencing parameters.yml
            tsfresh_features = TsfreshFe(
                df=outlet_df,
                date_column=constants["default_date_col"],
                target_feature=params_dict["tsfresh_target_feature"],
                days_per_group=params_dict["tsfresh_days_per_group"],
                bin_labels_list=params_dict["binning_dict"][target_feature],
                tsfresh_features_list=params_dict["tsfresh_features_list"],
                shift_period=params_dict["sma_tsfresh_shift_period"],
                n_significant=params_dict["tsfresh_n_significant"],
                num_features=params_dict["tsfresh_num_features"],
                relevant_features_dict=None,
            )

            relevance_table = tsfresh_features.generate_relevance_table()

            # Consolidate outlet based relevance table per fold
            if isinstance(relevance_table, pd.DataFrame):
                logger.info(
                    f"Relevance table for {outlet} has length of {len(relevance_table)}"
                )
                # Append relevance table for each outlet of a specific fold
                relevance_table_list.append(relevance_table)
            else:
                # No relevant features
                logger.info(
                    f"No relevance table for {outlet} found. Processing next outlet if available.\n"
                )
        if relevance_table_list:
            # To store fold based outlet relevant features
            relevant_features_dict = get_common_outlet_relevant_features(
                relevance_table_list, params_dict
            )
        else:
            logger.info(f"No relevant features generated for {fold}")
            relevant_features_dict = {}
        # Save as artefacts dictionary for current fold
        output_name = f"{fold}_tsfresh_relevant_features"
        artefact_dict[output_name] = relevant_features_dict
        logger.info(f"Completed feature selection for {fold}.\n")
    return artefact_dict


# 2nd node
def run_tsfresh_feature_engineering_process(
    partitioned_tsfresh_artefacts_input: Dict[str, pd.DataFrame],
    partitioned_outlet_input: Dict[str, pd.DataFrame],
    params_dict: Dict[str, Any],
) -> Dict:
    """Function which performs tsfresh feature engineering involving tsfresh feature creation for each outlet data in each fold based on learnt tsfresh artefacts and outlet inputs.

    Args:
        partitioned_tsfresh_artefacts_input (Dict[str, pd.DataFrame]): partitioned input for tsfresh relevant features json artefacts
        partitioned_outlet_input (Dict[str, pd.DataFrame): dictionary containing a fold_outlet as keys and outlet dataframes as values. E.g {fold1_outlet_1: outlet_1_df}
        params_dict (Dict[str, Any]): Parameters referencing parameters.yml.

    Returns:
        Dict: Dictionary which contains merged training and validation folds (E.g merged fold 1, merged fold 2 ..) as keys, which contain dictionaries of extracted tsfresh features by outlet.
    """
    fold_list = [key.split("_")[0] for key in partitioned_tsfresh_artefacts_input]

    # Instantiate empty dictionary serving as an output which stores
    extracted_features_dict = {}
    # iterate over folds
    for fold in fold_list:
        logger.info(f"Start tsfresh feature engineering for {fold}")
        # filter input by fold information corresponding fold outlet files
        outlet_list = [outlet for outlet in partitioned_outlet_input if fold in outlet]
        outlet_dict = {
            outlet: partitioned_outlet_input[outlet] for outlet in outlet_list
        }

        # Extract the correct tsfresh artefact id by fold info and read off the contents as dictionary
        params_key = [
            artefact_fold_id
            for artefact_fold_id in partitioned_tsfresh_artefacts_input
            if fold in artefact_fold_id
        ]
        relevant_features_dict = partitioned_tsfresh_artefacts_input[params_key[0]]

        # Create tsfresh features using relevant_features_dict and oultet_dict through function call 'feature_engineering_by_outlet' below
        outlet_features_dict = feature_engineering_by_outlet(
            relevant_features_dict=relevant_features_dict,
            outlet_dict=outlet_dict,
            params_dict=params_dict,
        )
        extracted_features_dict.update(outlet_features_dict)
    logger.info("Completed tsfresh feature engineering procedure.\n")
    return extracted_features_dict


def feature_engineering_by_outlet(
    relevant_features_dict: Dict[str, Dict],
    outlet_dict: Dict[str, pd.DataFrame],
    params_dict: Dict[str, Any],
) -> Dict[str, pd.DataFrame]:
    """Function which executes tsfresh feature engineering by outlet, extracting only the relevant features which are unique to each fold.

    Args:
        relevant_features_dict (Dict[str, Dict]): Dictionary containing the relevant calculated/learned tsfresh features artefacts
        outlet_dict (Dict[str, pd.DataFrame]): Kedro MemoryDataSet dictionary partitioned by outlets loaded in memory containing individual outlet dataframes.
        params_dict (Dict[str, Any]): Parameters referenced from parameters.yml.
        extracted_tsfresh_features_outlet_dict: Dict[str, pd.DataFrame]: Dictionary used for storing tsfresh_features_engineered identified by fold and outlet info.

    Returns:
        Dict[str, pd.DataFrame]: Dictionary storing outlet based generated tsfresh features in a dataframe using same outlet information as an identifier.
    """
    target_feature = params_dict["fe_target_feature_name"]  # Raw target column

    extracted_features_dict = {}
    for outlet in outlet_dict.keys():
        logger.info(f"Implementing tsfresh feature engineering for outlet {outlet}")
        outlet_df = outlet_dict[outlet]
        # Extract relevant features
        tsfresh_features = TsfreshFe(
            df=outlet_df,
            date_column=constants["default_date_col"],
            target_feature=params_dict["tsfresh_target_feature"],
            days_per_group=params_dict["tsfresh_days_per_group"],
            bin_labels_list=params_dict["binning_dict"][target_feature],
            tsfresh_features_list=params_dict["tsfresh_features_list"],
            shift_period=params_dict["sma_tsfresh_shift_period"],
            n_significant=params_dict["tsfresh_n_significant"],
            num_features=params_dict["tsfresh_num_features"],
            relevant_features_dict=relevant_features_dict,
        )
        tsfresh_df = tsfresh_features.extract_features()
        tsfresh_df = tsfresh_features.process_extracted_tsfresh_features_index(
            df=tsfresh_df
        )

        # drop rows with nan values
        tsfresh_df.dropna(axis=0, how="any", inplace=True)
        # rename tsfresh generated columns by removing characters " " and double underscore. e.g(lag_2_days_proxy_revenue__fft_coefficient__attr_"angle"__coeff_1)
        tsfresh_df.columns = [
            (re.sub(r"[^a-zA-Z0-9_,]", "", col)) for col in tsfresh_df.columns
        ]
        tsfresh_df.columns = [
            col.replace("__", "_").replace(",", "_") for col in tsfresh_df.columns
        ]

        # Create a custom identifier with outlet info to store tsfresh features.
        extracted_features_dict[outlet] = tsfresh_df
    return extracted_features_dict


def get_common_outlet_relevant_features(
    relevance_table_list: List[Union[pd.DataFrame, None]], params_dict: Dict[str, Any]
) -> Dict[str, float]:
    """Function which combines multiple outlets' relevance tables

    Args:
        relevance_table_list (List[Union[pd.DataFrame, None]]): List of relevance tables in pd.DataFrame.
        num_features (int): Number of top relevant features to keep based on derived sorted p-value vector from tsfresh's Benjamini-Yekutieli procedure.

    Returns:
        Dict[str, float]: Dictionary of relevant features that is in a suitable format for tsfresh extract features.
    """
    logger.info(f"Relevant features generated for {len(relevance_table_list)} outlets.")
    target_feature = params_dict["fe_target_feature_name"]  # Raw target column
    tsfresh_features = TsfreshFe(
        df=pd.DataFrame(),
        date_column=constants["default_date_col"],
        target_feature=params_dict["tsfresh_target_feature"],
        days_per_group=params_dict["tsfresh_days_per_group"],
        bin_labels_list=params_dict["binning_dict"][target_feature],
        tsfresh_features_list=params_dict["tsfresh_features_list"],
        shift_period=params_dict["sma_tsfresh_shift_period"],
        n_significant=params_dict["tsfresh_n_significant"],
        num_features=params_dict["tsfresh_num_features"],
        relevant_features_dict=None,
    )
    # For case of only single relevance table, take as it is, otherwise use tsfresh combine_relevance_tables
    if len(relevance_table_list) == 1:
        combined_relevance_table = relevance_table_list[0]
        combined_relevance_table.sort_values("p_value", inplace=True)
        combined_relevance_table = combined_relevance_table.iloc[
            : params_dict["tsfresh_num_features"]
        ]
    else:
        combined_relevance_table = tsfresh_features.combine_relevance_table(
            relevance_table_list
        )
    # Retrieve values of combined relevance tables
    features = combined_relevance_table["feature"].values
    logger.info(f"{len(features)} relevant features selected after combination.")
    kind_to_fc_parameters = tsfresh.feature_extraction.settings.from_columns(features)

    return kind_to_fc_parameters


def merge_tsfresh_features_with_outlets(
    partitioned_outlet_input: Dict[str, pd.DataFrame],
    partitioned_tsfresh_input: Dict[str, pd.DataFrame],
) -> Dict[str, pd.DataFrame]:
    """Function which left merges a outlet-based generated tsfresh features to each outlet based on common time-index. As the id used for both partitioned inputs are the same, joining can be simply made by using one of the id retreived from either of the inputs and doing a direct referencing from both dictionary keys.

    Args:
        partitioned_outlet_input: Dict[str, pd.DataFrame]: Kedro MemoryDataSet dictionary partitioned by outlets loaded in memory containing outlet dataframes.
        partitioned_mmm_input: Dict[str, pd.DataFrame]: Kedro IncrementalDataset of dataframe representing fold-based outlets.

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
        tsfresh_fold_df = partitioned_tsfresh_input[partitioned_id]
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
