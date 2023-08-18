from pydantic import BaseSettings, Field
from typing import List
import os

# get filepath to data_loader
current_directory = os.path.dirname(os.path.abspath(__file__))  # current path
parent_directory = os.path.dirname(os.path.dirname(current_directory))  # src
model_path = os.path.join(parent_directory, "models", "orderedmodel_prob_20230816.pkl")
log_conf_path = os.path.join(parent_directory, "conf", "base", "logging_inference.yml")


class Settings(BaseSettings):
    """
    Defines the application settings.

    This class uses pydantic's BaseSettings for environment variable parsing.

    Attributes:
        API_NAME (str): The name of the API, default is "BIPO FastAPI".
        API_VERSION (str): The version of the API, default is "/api/v1".
        LOGGER_CONFIG_PATH (str): The path to the logging configuration file.
        PRED_MODEL_UUID (str): The UUID of the prediction model.
        PRED_MODEL_PATH (str): The path to the prediction model file.
        INTERMEDIATE_OUTPUT_PATH (str): The path to output intermediate files.
        SALES_CLASS_NAMES (List[str]): The names of the sales classes.
    """

    API_NAME: str = Field(default="BIPO FastAPI", env="API_NAME")
    API_VERSION: str = Field(default="/api/v1", env="API_VERSION")
    LOGGER_CONFIG_PATH: str = Field(default=log_conf_path, env="LOGGER_CONFIG_PATH")
    PRED_MODEL_UUID: str = Field(default="0.1", env="PRED_MODEL_UUID")
    PRED_MODEL_PATH: str = Field(
        default=model_path,
        env="PRED_MODEL_PATH",
    )
    INTERMEDIATE_OUTPUT_PATH: str = Field(
        default="../data/10_model_inference_output", env="INTERMEDIATE_OUTPUT_PATH"
    )

    # The class names for sales classes, used in prediction responses as label to each class
    SALES_CLASS_NAMES: List[str] = [
        "Low",
        "Medium",
        "High",
        "Exceptional",
    ]


# Create an instance of the Settings class
SETTINGS = Settings()
