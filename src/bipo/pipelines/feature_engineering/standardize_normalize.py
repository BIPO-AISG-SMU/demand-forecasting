# Python script that contains series of functions involving standardisation/normalisation methods
from typing import Union, List, Tuple, Dict
import pandas as pd
import numpy as np

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
    df: pd.DataFrame, std_norm_object: Union[Normalizer, StandardScaler]
) -> pd.DataFrame:
    """Function which utilises sklearn learned standardscaler/normalizer object 'std_norm_object' containing list of features for value transformation which can be found in dataframe columns assuming same dataframe is used.

    Args:
        df (pd.DataFrame): Dataframe to process for standardisation/normalisation.
        std_norm_object (Union[Normalizer, StandardScaler]): Learned sklearn Normalizer or StandardScaler object.

    Raises:
        None.

    Returns:
        pd.DataFrame: Updated dataframe with normalized values in features selected.
    """
    feature_names_list = std_norm_object.feature_names_in_

    std_norm_features = std_norm_object.transform(df[feature_names_list])

    # Update dataframe for standardised/normalized features inplace
    std_normed_df = pd.DataFrame(
        std_norm_features, columns=feature_names_list, index=df.index
    )

    df.update(std_normed_df)

    return df
