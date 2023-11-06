import pandas as pd
import os
from kedro.config import ConfigLoader
from kedro.io import DataCatalog
from kedro_datasets.json import JSONDataSet
from kedro_datasets.pickle import PickleDataSet
from kedro_datasets.pandas import CSVDataSet
from bipo import settings
import logging

LOGGER = logging.getLogger(settings.LOGGER_NAME)
# Instantiate config
CONF_LOADER = ConfigLoader(conf_source=settings.CONF_SOURCE)


def load_artefact(artefact_filepath: str) -> dict:
    """Loads json/pickle artefact dictionary from specified filepath

    Args:
        artefact_filepath (str): Directory path of the artefact

    Returns:
        dict: Artefact dictionary
    """
    if len(os.listdir(artefact_filepath)) < 1:
        LOGGER.error(f"No artefact in {artefact_filepath}")
    elif len(os.listdir(artefact_filepath)) > 1:
        LOGGER.error(f"More than 1 artefact in {artefact_filepath}")
    else:
        LOGGER.info(f"Load artefact from {artefact_filepath}")
        for filepath in os.listdir(artefact_filepath):
            filepath = os.path.join(artefact_filepath, filepath)
            if filepath.endswith(".json"):
                artefact = JSONDataSet(filepath=filepath)
            elif filepath.endswith(".pkl"):
                artefact = PickleDataSet(filepath=filepath)
            return artefact.load()


# Load data
def load_data_catalog() -> DataCatalog:
    """Load data files and artefacts and add them into a data catalog

    Returns:
        DataCatalog: Catalog containing all necessary data files
    """
    conf_inference = CONF_LOADER.get("inference*")
    conf_const = CONF_LOADER.get("constants*")
    conf_params = CONF_LOADER["parameters"]
    mkt_df = CSVDataSet(filepath=conf_const["inference"]["marketing_filepath"]).load()
    outlet_df = CSVDataSet(
        filepath=conf_const["inference"]["outlet_filepath"], load_args={"index_col": 0}
    ).load()
    lag_sales_df = CSVDataSet(
        filepath=conf_const["inference"]["lag_sales_filepath"],
        load_args={"index_col": 0},
    ).load()

    # Convert to datetime index
    outlet_df.index = pd.to_datetime(outlet_df.index)
    lag_sales_df.index = pd.to_datetime(lag_sales_df.index)
    # Cost_centre_code to get the correct encoding and std_norm artefact
    cost_centre_code = outlet_df.pop("cost_centre_code")[0]
    # partition dataset
    tsfresh_relevant_features = load_artefact(conf_const["inference"]["tsfresh_relevant_features_filepath"])

    # Load specific fold encoding artefact
    std_norm_artefact = load_artefact(conf_const["inference"]["std_norm_filepath"])
    std_key = f'{conf_inference["artefact_fold"]}_{cost_centre_code}_standardize'
    norm_key = f'{conf_inference["artefact_fold"]}_{cost_centre_code}_normalize'
    if std_key in std_norm_artefact.keys():
        std_norm_artefact = std_norm_artefact[std_key]
    elif norm_key in std_norm_artefact.keys():
        std_norm_artefact = std_norm_artefact[norm_key]
    else:
        LOGGER.info(
            "No standardization or normalization key is found in std_norm.pkl artefact"
        )
        std_norm_artefact = 0

    # Encoding
    encoding_artefact = load_artefact(conf_const["inference"]["encoding_filepath"])
    ohe_key = f'{conf_inference["artefact_fold"]}_ohe'
    ordinal_encoding_key = f'{conf_inference["artefact_fold"]}_ord'
    if ohe_key in encoding_artefact.keys():
        ohe_artefact = encoding_artefact[ohe_key]
    else:
        LOGGER.info("One_hot_encoding key is not found in encoding.pkl artefact")
        ohe_artefact = 0
    if ordinal_encoding_key in encoding_artefact.keys():
        ordinal_encoding_artefact = encoding_artefact[ohe_key]
    else:
        LOGGER.info("Ordinal_encoding key is not found in encoding.pkl artefact")
        ordinal_encoding_artefact = 0

    # Populate data catalog
    catalog = DataCatalog()
    catalog.add_feed_dict(
        {
            "parameters": conf_params,
            "mkt_df": mkt_df,
            "lag_sales_df": lag_sales_df,
            "outlet_df": outlet_df,
            "std_norm_artefact": std_norm_artefact,
            "ohe_artefact": ohe_artefact,
            "ordinal_encoding_artefact": ordinal_encoding_artefact,
            "tsfresh_relevant_features": tsfresh_relevant_features,
        },
        replace=True,
    )
    return catalog
