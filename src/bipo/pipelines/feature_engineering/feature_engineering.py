import pandas as pd
import numpy as np
import argparse
import openpyxl
import json
from datetime import date
from typing import Union, List, Tuple, Dict
import os
from kedro.config import ConfigLoader
from kedro.io import DataCatalog
from .endo import Endogenous
from .exog import Exogenous
from .tsfresh_fe import TsfreshFe
from .common import read_csv_file
from bipo.utils import get_input_output_folder, get_project_path

import logging

logging = logging.getLogger(__name__)

# Instantiate config
project_path = get_project_path()
conf_loader = ConfigLoader(conf_source=project_path / "conf")
conf_params = conf_loader["parameters"]
constants = conf_loader.get("constants*")
conf_catalog = conf_loader.get("catalog*", "catalog/**")
catalog = DataCatalog.from_config(conf_catalog)

# load configs
FEATURE_ENGINEERING_DROP_COLUMN_LIST = constants["feature_engineering_drop_column_list"]


def run_fe_pipeline(outlet_df: pd.DataFrame) -> Tuple[pd.DataFrame, list, list]:
    """Function that integrates endogenous and exogenous features, concatenates them and removes any duplicate columns.

    Args:
        None

    Raises:
        None

    Returns:
        Tuple[pd.DataFrame, list, list] containing:
        - df_integrated: The integrated DataFrame comprising all endogenous and exogenous features engineered or non-engineered.
        - added_exogenous feature list: List containing added exogenous features
        - added_endogenous feature list: List containing added endogenous features
    """
    # Engineer endogenous features using class import followed by its endo_transform method
    logging.info("Starting feature engineering of endogenous features")
    endo_df = Endogenous(outlet_df)
    endo_df_tf = endo_df.endo_transform()
    # early termination if cannot bin target feature
    if endo_df_tf is None:
        return ("empty", [], [])

    # Get endogenous features artefacts
    added_endo_feature_list = endo_df.get_added_features_endo()

    # Engineer exogenous features using class import followed by its exog_transform method.
    logging.info("Starting feature engineering of exdogenous features")
    exog_df = Exogenous(outlet_df)
    exog_df_tf = exog_df.exog_transform()

    # Get exogenous features artefacts
    added_exog_feature_list = exog_df.get_added_features_exog()

    # Concatenate all dataFrames (include endo/exo features engineered) horizontally with original data.
    logging.info("Concatenating all features and removing duplicated features")

    # Concatenation is guaranteed with the inclusion of self.merged_df, which is never empty. Minimally, self.merged_df will be the subdataframe that is included

    logging.debug(
        f"Merging exogenous {endo_df_tf.shape}, endogenous {exog_df_tf.shape} and original dataframes {outlet_df.shape}"
    )

    df_integrated = outlet_df.copy()
    df_integrated.index = pd.to_datetime(df_integrated.index)
    # Merge with endogenous variable and exogenous variable
    df_merge_list = [endo_df_tf, exog_df_tf]

    for df in df_merge_list:
        df.index = pd.to_datetime(df.index)
        df_integrated = df_integrated.merge(
            df,
            how="left",
            left_index=True,
            right_index=True,
            suffixes=("", "_remove"),
        )
        duplicate_col_list = [col for col in df_integrated.columns if "_remove" in col]
        logging.debug(f"Removing {len(duplicate_col_list)} duplicated columns")
        df_integrated.drop(duplicate_col_list, axis=1, inplace=True)
    logging.debug("Removing unnecessary columns")
    df_integrated.drop(FEATURE_ENGINEERING_DROP_COLUMN_LIST, axis=1, inplace=True)
    # set date index as column
    df_integrated.reset_index(inplace=True)
    return df_integrated, added_exog_feature_list, added_endo_feature_list


def save_artefacts(
    added_exog_feature_list: list,
    added_endo_feature_list: list,
) -> None:
    """Save engineered exogenous and endogenous features as a json file artefact using the catalog.

    Args:
        added_exog_feature_list (list): list of exogenous engineered feature names.
        added_endo_feature_list (list): list of endogenous engineered feature names.

    Returns:
        None
    """
    # Dictionary to store added feature names as part of feature engineering
    artefacts_dict = {
        "Added Endogenous": added_endo_feature_list,
        "Added Exogenous": added_exog_feature_list,
    }
    catalog.save("engineered_features_list", artefacts_dict)


def run_tsfresh_fe_pipeline(df_integrated: pd.DataFrame, params) -> pd.DataFrame:
    """performs tsfresh feature extraction on dataframe that has endogenous and exogenous engineered features.

    Args:
        df_integrated (pd.DataFrame): dataframe containing endogenous and exogenous engineered features.
        params (dict): parameters config

    Returns:
        DataFrame: dataframe containing tsfresh extracted features.
    """
    logging.info("Starting feature engineering using tsfresh")
    tsfresh_features = TsfreshFe(
        df_integrated, params["feature_engineering"]["endo"]["tsfresh_entity"]
    )
    tsfresh_df = tsfresh_features.run_fe_pipeline(
        params["feature_engineering"]["endo"]["extract_relevant"],
        params["feature_engineering"]["endo"]["tsfresh_days_per_group"],
        # params["feature_engineering"]["endo"]["tsfresh_save_path"],
    )

    # rename tsfresh generated columns by removing the " ". e.g(lag_2_days_proxy_revenue__fft_coefficient__attr_"angle"__coeff_1)
    def process_column_name(column_name: str) -> str:
        """helper function to rename tsfresh generated features' column names by removing the " ", and replacing the double underscores with a single underscore

        Args:
            column_name (str): column name of tsfresh generated feature

        Returns:
            str: renamed column name
        """
        # Remove double quotes
        processed_name = column_name.replace('"', "")
        # Replace double underscores with a single underscore
        processed_name = processed_name.replace("__", "_")
        return processed_name

    tsfresh_df.columns = [process_column_name(col) for col in tsfresh_df.columns]

    # merge with df_integrated
    df_integrated = pd.merge(
        df_integrated,
        tsfresh_df,
        left_index=True,
        right_index=True,
        how="left",
    )
    # to get date column from index
    df_integrated = df_integrated.reset_index()
    # drop rows with nan values due to lagged features
    drop_rows_index_list = df_integrated[df_integrated.isnull().any(axis=1)].index
    df_integrated.drop(drop_rows_index_list, inplace=True)
    logging.info(
        f"Removing {len(drop_rows_index_list)} rows due to missing values as a result of lagged features."
    )
    return df_integrated


def run_tsfresh_feature_selection(
    relevance_table_list: list, df_integrated: pd.DataFrame, params
) -> list:
    """generates relevance table after endogenous and exogenous feature engineering, and stores the relevance table in a list.

    Args:
        df_integrated (DataFrame): dataframe containing all features, including endogenous and exogenous engineered features.
        relevance_table_list (list): list to hold relevance tables.

    Returns:
        list: list of relevance tables.
    """
    # filter the dataframe as extracting all possible features for the entire dataframe across multiple outlets will cause kernel to fail.
    df_integrated = df_integrated.iloc[
        : params["feature_engineering"]["endo"]["filter_num_days"]
    ]
    tsfresh_features = TsfreshFe(
        df_integrated, params["feature_engineering"]["endo"]["tsfresh_entity"]
    )
    relevance_table = tsfresh_features.generate_relevance_table(
        days_per_group=params["feature_engineering"]["endo"]["tsfresh_days_per_group"],
        n_significant=params["feature_engineering"]["endo"]["n_significant"],
    )
    relevance_table_list.append(relevance_table)
    return relevance_table_list


def save_tsfresh_relevant_features(relevance_table_list: list, params) -> None:
    """saves tsfresh relevant features

    Args:
        relevant_table_list (list): list of relevance tables.

    Returns:
        None
    """
    if all(table is None for table in relevance_table_list):
        logging.info("No relevant features found.")
    else:
        # initialize class without parameters
        tsfresh_features = TsfreshFe("blank", "blank")
        combined_relevance_table = tsfresh_features.combine_relevance_table(
            relevance_table_list,
            params["feature_engineering"]["endo"]["tsfresh_num_features"],
        )
        tsfresh_features.save_relevant_features(combined_relevance_table)
