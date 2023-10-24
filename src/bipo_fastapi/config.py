from pydantic import BaseSettings, Field
from typing import List

from kedro.config import ConfigLoader
from kedro.framework.project import settings
# initiate to inference.yml and parameters.yml
conf_loader = ConfigLoader(conf_source=settings.CONF_SOURCE)
conf_inference_files = conf_loader.get("inference*")
conf_params = conf_loader["parameters"]

# output files:
INTERMEDIATE_OUTPUT_PATH = conf_inference_files["file_paths"][
    "intermediate_output_path"
]

# bin list (get from params, but will be changed after refactor)
TARGET = conf_params["fe_target_feature_name"]
BIN_LABEL_LIST = conf_params["binning_dict"][TARGET]


# path to model weights
MODEL_PATH = conf_inference_files["file_paths"]["path_to_model"]


class Settings(BaseSettings):
    """
    Defines the application settings.

    This class uses pydantic's BaseSettings for environment variable parsing.

    Attributes:
        API_NAME (str): The name of the API, default is "BIPO FastAPI".
        API_VERSION (str): The version of the API, default is "/api/v1".
        PRED_MODEL_UUID (str): The UUID of the prediction model.
        PRED_MODEL_PATH (str): The path to the prediction model file.
        INTERMEDIATE_OUTPUT_PATH (str): The path to output intermediate files.
        SALES_CLASS_NAMES (List[str]): The names of the sales classes.
    """

    API_NAME: str = Field(default="BIPO FastAPI", env="API_NAME")
    API_VERSION: str = Field(default="/api/v1", env="API_VERSION")
    PRED_MODEL_UUID: str = Field(default="0.1", env="PRED_MODEL_UUID")
    PRED_MODEL_PATH: str = Field(
        default=MODEL_PATH,
        env="PRED_MODEL_PATH",
    )
    INTERMEDIATE_OUTPUT_PATH: str = Field(
        default=INTERMEDIATE_OUTPUT_PATH, env="INTERMEDIATE_OUTPUT_PATH"
    )

    # The class names for sales classes, used in prediction responses as label to each class
    SALES_CLASS_NAMES: List[str] = BIN_LABEL_LIST


# Create an instance of the Settings class
SETTINGS = Settings()
