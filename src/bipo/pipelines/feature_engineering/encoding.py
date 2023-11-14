# Python script that contains series of functions involving encoding related functions.

from typing import Union, List, Dict, Any, Type
import pandas as pd
import numpy as np

from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from bipo import settings
import logging

logger = logging.getLogger(settings.LOGGER_NAME)


def ordinal_encoding_fit(
    df: pd.DataFrame,
    ordinal_columns_dict: Dict[str, List],
    ordinal_encoding_dict: Dict[str, OrdinalEncoder],
    fold: str,
) -> Dict[str, OrdinalEncoder]:
    """Function which only implements ordinal encoding fitting to generate parameters for transform. Assumes the ordinal_columns_dict contains ordinal columns which are found in dataframe to be processed.

    Args:
        df (pd.DataFrame): DataFrame to be processed.
        ordinal_columns_dict (Dict[str, List]): Dict containing keys indicating columns to be ordinal encoded and its correspondign ordinal encoding categories in a list.
        ordinal_encoding_dict (Dict[str, OrdinalEncoder]): Loaded object in dictionary by Kedro containing ordinal encoding by <fold>_<outlet> as key and encoding information as value.
        fold (str): Fold information which dataframe is processed.

    Raises:
        None.

    Returns:
        Dict[str, OrdinalEncoder]: Updated dictionary containing ordinal encoding information for specific fold and outlet.
    """

    ordinal_col_list = list(ordinal_columns_dict.keys())

    logger.info(
        f"Number of features to be ordinal encoded: {len(ordinal_columns_dict)}"
    )

    ordinal_col_set = set(ordinal_col_list)
    # Get object type columns (numerical values typecasted in string) which may be categorical
    df_column_set = set(df.columns)

    # Find valid columns referencing dataframe:
    corrected_ordinal_col_list = list[(df_column_set).intersection(ordinal_col_set)]

    # Get the relevant values from
    ordinal_values_list = [
        list(ordinal_columns_dict[col]) for col in corrected_ordinal_col_list
    ]
    df[corrected_ordinal_col_list] = df[corrected_ordinal_col_list].astype(str)

    logger.info(
        f"Number of features to be ordinal encoded: {len(corrected_ordinal_col_list)} after correction"
    )

    logger.info("Applying ordinal encoding on training data")

    ord_encoder = OrdinalEncoder(
        categories=ordinal_values_list,
        handle_unknown="use_encoded_value",
        unknown_value=-1,  # For unknown categories
        encoded_missing_value=-1,  # For missing values
    )

    logger.info(f"Applying ordinal fit valid columns: {corrected_ordinal_col_list}")
    # fit other columns (artefacts saved)
    ord_encoder.fit(df[corrected_ordinal_col_list])

    ordinal_encoding_dict[f"{fold}_ord"] = ord_encoder
    return ordinal_encoding_dict


def ordinal_encoding_transform(
    df: pd.DataFrame,
    ordinal_encoder: OrdinalEncoder,
) -> pd.DataFrame:
    """Function which only implements ordinal encoding transformation based on an encoding dictionary containing encodings for respective fold-based concatenated outlet dataframe.

    Args:
        df (pd.DataFrame): DataFrame to be processed.
        ordinal_encoder (OrdinalEncoder): Fitted sklearn OrdinalEncoder.


    Raises:
        None

    Returns:
        pd.DataFrame: A ordinal encoded dataframe.
    """
    # Retrieve feature names seen during fit by ordinalencoder
    ordinal_features_list = ordinal_encoder.feature_names_in_

    logger.info(f"Applying ordinal encoding with features: {ordinal_features_list}")

    # Create new columns representing ordinal encoded feature
    try:
        new_ordinal_feature = [f"ord_{feature}" for feature in ordinal_features_list]
        df[new_ordinal_feature] = ordinal_encoder.transform(df[ordinal_features_list])

    except KeyError:
        logger.error(
            f"Unable to find either feature specified in {ordinal_features_list} in dataframe. Ordinal encoding would not be implemented."
        )
    return df


def one_hot_encoding_fit(
    df: pd.DataFrame,
    ohe_column_list: List,
    ohe_encoding_dict: Dict[str, OneHotEncoder],
    fold: str,
) -> Dict:
    """Function which only implements ordinal encoding transformation based on a encoding dictionary containing encodings for respective fold/outlet dataframe.

    Args:
        df (pd.DataFrame): DataFrame to be processed.
        ohe_column_list (List): List of columns to be one hot encoded.
        ohe_encoding_dict (Dict[str, OneHotEncoder]): Loaded object in dictionary by Kedro containing learned one-hot encoding using '<fold>_<outlet>' as key and encoding information as value.
        fold (str): Fold information which dataframe is processed.

    Raises:
        None

    Returns:
        pd.DataFrame: A ordinal encoded dataframe.
    """
    # Check for cases when empty
    logger.info(f"Number of features to be onehotencoded: {len(ohe_column_list)}")

    ohe_column_set = set(ohe_column_list)
    # Get object type columns (numerical values typecasted in string) which may be categorical
    df_column_set = set(df.columns)

    # Find common columns:
    corrected_ohe_column_list = list((df_column_set).intersection(ohe_column_set))

    logger.info(
        f"Possible columns to encode after validating their existence: {corrected_ohe_column_list}"
    )

    if not corrected_ohe_column_list:
        logger.info(
            "No valid columns to encode after validation. No updates to be made to encoding parameters dictionary."
        )
    else:
        logger.info(
            f"Instantiating sklearn's OneHotEncoder and dropping first instance."
        )
        ohe_encoder = OneHotEncoder(
            categories="auto",
            handle_unknown="ignore",
            drop="first",
            sparse_output=False,
        ).fit(df[corrected_ohe_column_list])
        ohe_encoding_dict[f"{fold}_ohe"] = ohe_encoder

        logger.info(f"Saved one hot encodings learned for {fold}.\n")
    return ohe_encoding_dict


def one_hot_encoding_transform(
    df: pd.DataFrame,
    ohe_encoder: OneHotEncoder,
) -> pd.DataFrame:
    """Function which applies retrieves one-hot-encoding on a provided dictionary of one-hot-encodings based on fold-based concatenated outlet dataframe and applies a transform. This is based on the assumption that all columns provided exists in the dataframe.

    Args:
        df (pd.DataFrame): Dataframe to be processed.
        ohe_encoder (OneHotEncoder): Fitted sklearn OneHotEncoder.
    Raises:
        None

    Returns:
        pd.DataFrame: Dataframe with encoded columns.
    """

    try:
        ohe_features_list = ohe_encoder.feature_names_in_
        ohe_feature_names = ohe_encoder.get_feature_names_out(ohe_features_list)
        logger.info(f"One-hot encoded feature names: {ohe_feature_names}")
        # Apply transform using learned features
        ohe_df = pd.DataFrame(
            ohe_encoder.transform(df[ohe_features_list]),
            index=df.index,
            columns=ohe_feature_names,
        )
        # Drop onehotencoded colums and replace with new encoded columns
        df.drop(columns=ohe_features_list, inplace=True)
        df = pd.concat([df, ohe_df], axis=1)
    except KeyError:
        logger.error(
            f"Unable to find either feature specified in {ohe_features_list} in dataframe. One-hot encoding would not be implemented."
        )
    return df
