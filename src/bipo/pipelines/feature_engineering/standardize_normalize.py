# Python script that contains series of functions involving standardisation/normalisation methods
from typing import Union, List, Tuple, Dict
import pandas as pd
import numpy as np
import sys
import os
import json
import pickle

from kedro.config import ConfigLoader
from sklearn.preprocessing import Normalizer, StandardScaler
from bipo import settings
import logging

# This will get the active logger during run time
logger = logging.getLogger(settings.LOGGER_NAME)
conf_loader = ConfigLoader(settings.CONF_SOURCE)

def standard_norm_fit(
    df: pd.DataFrame,
    col_to_std_or_norm_list: List,
    std_norm_encoding_dict: Dict,
    option: str,
    fold: str,
    outlet: str,
) -> Dict:
    """Function which applies sklearn standardscaler/normalizer fit to learn parameters required for normalization and updates a input dictionary that contains fold and outlet informations' normalization parameters. It assumes that the col_to_std_or_norm_list contains columns which can be found in dataframe columns. Assumes col_to_std_or_norm_list is never empty.

    Args:
        df (pd.DataFrame): Dataframe to process.
        col_to_std_or_norm_list (List): List of dataframe columns which standardization is to be applied.
        std_norm_encoding_dict (Dict): Dictionary that contains fold and outlet informations' standardization/normalisation parameters which is called by nodes.py.
        fold (str): fold information which the dataframe represents.
        outlet (str): outlet information which the dataframe represents.

    Raises:
        None.

    Returns:
        Dict: Updated dictionary that contains fold and outlet informations' standardization parameters.
    """
    if option.lower() == "standardize":
        logger.info(f"Applying sklearn standardscaler to {col_to_std_or_norm_list}")
        std_norm = StandardScaler()

    # Otherwise default to normalize
    else:
        logger.info(f"Applying sklearn normalizer to {col_to_std_or_norm_list}")
        option = "normalize"
        std_norm = Normalizer()

    # Apply fit based on instantiated normalizer
    fitted_params = std_norm.fit(df[col_to_std_or_norm_list])

    # Update dictionary for learned parameters
    std_norm_encoding_dict[f"{fold}_{outlet}_{option}"] = fitted_params
    return std_norm_encoding_dict


def standard_norm_transform(
    df: pd.DataFrame,
    col_to_std_or_norm_list: List,
    std_norm_encoding_dict: Dict,
    option: str,
    fold: str,
    outlet: str,
) -> pd.DataFrame:
    """Function which applies learned sklearn standardscaler/normalizer parameters learned from a dictionary that contains fold and outlet informations' standard scaling parameters. It assumes that the col_to_std_or_norm_list contains columns which can be found in dataframe columns.

    Args:
        df (pd.DataFrame): Dataframe to process.
        col_to_std_or_norm_list (List): List of dataframe columns which standardization is to be applied.
        std_norm_encoding_dict (Dict): Dictionary that contains fold and outlet informations' standardization/normalisation parameters.
        fold (str): fold information which the dataframe represents.
        outlet (str): outlet information which the dataframe represents.

    Raises:
        None.

    Returns:
        pd.DataFrame: Updated dataframe with normalized values in features selected.
    """
    if option.lower() != "standardize":
        # automatically override to normalize to ensure proper key referencing.
        option = "normalize"

    std_norm = std_norm_encoding_dict[f"{fold}_{outlet}_{option}"]

    std_norm_features = std_norm.transform(df[col_to_std_or_norm_list])

    # Update dataframe for standardised/normalized features inplace
    std_normed_df = pd.DataFrame(
        std_norm_features, columns=col_to_std_or_norm_list, index=df.index
    )

    df.update(std_normed_df)

    return df
