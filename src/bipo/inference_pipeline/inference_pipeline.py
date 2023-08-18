from bipo.inference_pipeline.feature_engineering import FeatureEngineering
from bipo.inference_pipeline.data_preprocessing import PreprocessInferenceData
from bipo.inference_pipeline.model_specific_fe import ModelSpecificFE

import pandas as pd

# import logging
import logging.config
import yaml
import os

# from kedro.config import ConfigLoader
from bipo.utils import get_project_path

project_path = get_project_path()

yaml_file = os.path.join(project_path, "conf", "base", "logging_inference.yml")

with open(yaml_file, "r") as stream:
    config = yaml.load(stream, Loader=yaml.FullLoader)
    print(config)

# Apply changes
logging.config.dictConfig(config)

# LOGGING_NAME = constants["inference"]["logging_name"]
logging = logging.getLogger("inference")


# Runs inference datapipeline which includes preprocessing, feature engineering, model specific preprocessing and generates a dataframe to be used for model inference.
def run_pipeline(df: pd.DataFrame):
    """Instantiate and runs preprocessing steps, feature engineering steps and model specific feature engineering steps on the input DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame to be preprocessed.

    Returns:
        final_df(pd.DataFrame): Dataframe after preprocessing and feature engineering.
    """

    logging.info("Starting Preprocess and FE of inference Data")
    # instantiate the preprocess class
    preprocess_inference = PreprocessInferenceData(df)
    merged_df, preprocessed_lagged_sales_df = preprocess_inference.run_pipeline()

    # instantiate the feature engineering class
    feature_engineering = FeatureEngineering(merged_df, preprocessed_lagged_sales_df)
    df_integrated = feature_engineering.run_pipeline()

    # instantiate the Model specific feature engineering class
    model_preprocess = ModelSpecificFE(df_integrated)
    final_df = model_preprocess.run_pipeline()
    logging.info("Completed Preprocess and FE of inference Data")
    return final_df
