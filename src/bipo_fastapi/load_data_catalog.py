import pandas as pd
import os 
from kedro.config import ConfigLoader
from kedro.io import DataCatalog, PartitionedDataset
from kedro_datasets.json import JSONDataSet 
from kedro_datasets.pickle import PickleDataSet
from kedro_datasets.pandas import CSVDataSet
from bipo import settings
import logging
LOGGER = logging.getLogger(settings.LOGGER_NAME)
# instantiate config
conf_loader = ConfigLoader(conf_source=settings.CONF_SOURCE)
conf_inference = conf_loader.get("inference*")
conf_const = conf_loader.get("constants*")
conf_params = conf_loader["parameters"]

def load_artefact(artefact_filepath:str)->dict:
    """loads json/pickle artefact dictionary from specified filepath 

    Args:
        artefact_filepath (str): directory path of the artefact

    Returns:
        dict: artefact dictionary
    """
    if len(os.listdir(artefact_filepath)) < 1:
        logging.error(f"No artefact in {artefact_filepath}")
    elif len(os.listdir(artefact_filepath)) > 1:
        logging.error(f"More than 1 artefact in {artefact_filepath}")
    else: 
        logging.info(f"Extract fold from {artefact_filepath} to rename dataframe")
        for filepath in os.listdir(artefact_filepath):
            filepath = os.path.join(artefact_filepath, filepath)
            if filepath.endswith(".json"):
                artefact =  JSONDataSet(filepath=filepath)
            elif filepath.endswith(".pkl"):
                artefact = PickleDataSet(filepath=filepath)
            return artefact.load()

# load data 
def load_data_catalog()->DataCatalog:
    """load data files and artefacts and add them into a data catalog   

    Returns:
        DataCatalog: catalog containing all necessary data files
    """
    mkt_df=CSVDataSet(filepath=conf_const["inference"]["marketing_filepath"]).load()
    outlet_df = CSVDataSet(filepath=conf_const["inference"]["outlet_filepath"],load_args={"index_col":0}).load()
    lag_sales_df = CSVDataSet(filepath=conf_const["inference"]["lag_sales_filepath"],load_args={"index_col":0}).load()

    # convert to datetime index
    outlet_df.index = pd.to_datetime(outlet_df.index)
    lag_sales_df.index = pd.to_datetime(lag_sales_df.index)
    # cost_centre_code to get the correct encoding and std_norm artefact
    cost_centre_code = outlet_df.pop("cost_centre_code")[0]
    # partition dataset
    lightweightmmm_params =  load_artefact(conf_const["inference"]["lightweightmmm_params_filepath"])[conf_inference["artefact_fold"]]
    tsfresh_relevant_features = load_artefact(conf_const["inference"]["tsfresh_relevant_features_filepath"])

    # load specific fold encoding artefact 
    std_norm_artefact = load_artefact(conf_const["inference"]["std_norm_filepath"])
    std_norm_key = f'{conf_inference["artefact_fold"]}_{cost_centre_code}_standardize'
    if std_norm_key in std_norm_artefact.keys():
        std_norm_artefact = std_norm_artefact[std_norm_key]
    else:
        std_norm_key = f'{conf_inference["artefact_fold"]}_{cost_centre_code}_normalize'
        std_norm_artefact = std_norm_artefact[std_norm_key]
    
    # encoding 
    encoding_artefact = load_artefact(conf_const["inference"]["encoding_filepath"])
    ohe_key = f'{conf_inference["artefact_fold"]}_ohe'
    ordinal_encoding_key = f'{conf_inference["artefact_fold"]}_ord'
    ohe_artefact = {}
    if ohe_key in encoding_artefact.keys():
        ohe_artefact = encoding_artefact[ohe_key]
    ordinal_encoding_artefact = {}
    if ordinal_encoding_key in encoding_artefact.keys():
        ordinal_encoding_artefact = encoding_artefact[ohe_key]


    # populate data catalog
    catalog = DataCatalog()
    catalog.add_feed_dict(
        {
            "parameters": conf_params,
            "mkt_df": mkt_df,
            "lag_sales_df": lag_sales_df,
            "outlet_df": outlet_df,
            "lightweightmmm_params": lightweightmmm_params,
            "std_norm_artefact": std_norm_artefact,
            "ohe_artefact": ohe_artefact,
            "ordinal_encoding_artefact": ordinal_encoding_artefact,
            "tsfresh_relevant_features": tsfresh_relevant_features,
            "cost_centre_code": cost_centre_code
        },
        replace=True,
    )
    return catalog