import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import logging
import pickle
from typing import List, Union
from ast import literal_eval

# Create a logger for this module
LOGGER = logging.getLogger(__name__)

from bipo_fastapi.config import SETTINGS
from bipo_fastapi.schemas import SalesAttributes, SalesPredictions, SalesPrediction
from bipo.inference_pipeline.inference_pipeline import run_pipeline


class PredictionModel:
    """
    This class is used for making predictions with a trained model.

    Attributes:
        model : The trained model.
        class_names (List[str]) : The names of the sales classes.
    """

    def __init__(self, model, class_names: List[str]):
        if model is None or class_names is None:
            LOGGER.error("Both model and class_names need to be provided")
            return None

        self.model = model
        self.class_names = class_names
        LOGGER.info(
            f"Initialized PredictionModel: {model} and class_names: {class_names}"
        )

    def transform_model_input(self, sales_df):
        """
        Transform the API request sales data to model input data

        Returns:
            A DataFrame representation of the model input data.
        """
        # Suppose to call the actual data transformation pipeline

        # Load mockup data for now
        # model_input_data = pd.read_csv(
        #     r"C:\Users\myeng\Downloads\aiap\git_issues\135_data_transformation_pipeline_for_api_request_payload\src\bipo_fastapi\zSample_Model_Input.csv"
        # )
        model_input_data = run_pipeline(sales_df)
        if "date" in model_input_data.columns:
            model_input_data.set_index("date", inplace=True)
        return model_input_data

    def transform_attributes(self, sales_attrs: SalesAttributes) -> pd.DataFrame:
        """
        Converts SalesAttributes into DataFrames.

        Args:
            sales_attrs (SalesAttributes) : An instance of the SalesAttributes class.

        Returns:
            DataFrames representation of dates_cost_centre and sales_attrs.
        """
        # try:
        sales_data = [attr.dict() for attr in sales_attrs.sales_attributes]
        if len(sales_data):  # check if data is empty
            sales_df = pd.DataFrame(sales_data)
            if "date" in sales_df.columns:  # if date is in columns
                sales_df.set_index("date", inplace=True)

            LOGGER.info(f"Saving model input data.")
            output_intermediate_file(sales_df, "model_input")

            try:
                dates = sales_df.index
                cost_centre_codes = sales_df.pop("cost_centre_code")
                model_input_df = self.transform_model_input(sales_df)
                return dates, cost_centre_codes, model_input_df

            except (ValueError, KeyError, TypeError) as e:
                LOGGER.error(f"Failed to transform sales attributes: {e}")
                return None

    def predict(self, sales_attrs: SalesAttributes) -> SalesPredictions:
        """
        This method is used to predict the sales classes using the trained model.

        Args:
            sales_attrs (SalesAttributes) : An instance of the SalesAttributes class.

        Returns:
            SalesPredictions : An instance of the SalesPredictions class.
        """
        try:
            LOGGER.info(f"Saving API request data.")
            output_intermediate_file(sales_attrs, "API_request", "json")

            LOGGER.info(f"Transforming API request to model's expected format.")
            dates, cost_centre_codes, model_input_data = self.transform_attributes(
                sales_attrs
            )

            LOGGER.info(f"Executing model inference.")
            predictions = self.model.model.predict(
                self.model.params, exog=model_input_data
            )

            # Uncomment this line below to test 500 Internal Server Error with empty fields
            # predictions = None

            LOGGER.info(f"Raw predictions generated: {predictions}")

            if predictions is not None:
                LOGGER.info(f"Formatting predictions for API response.")
                # Format the predictions into SalesPredictions class
                sales_preds = SalesPredictions(
                    sales_predictions=[
                        SalesPrediction(
                            date=date,
                            cost_centre_code=cost_centre_code,
                            sales_class_id=str(class_id := pred.argmax()),
                            sales_class_name=self.class_names[class_id],
                            probabilities={
                                str(i): f"{p:.4f}" for i, p in enumerate(pred)
                            },
                        )
                        for date, cost_centre_code, pred in zip(
                            dates, cost_centre_codes, predictions
                        )
                    ]
                )
            else:
                LOGGER.error(f"The model returned no predictions.")
                return None

            LOGGER.info(f"Saving API response data.")
            output_intermediate_file(sales_preds, "API_response", "json")

            LOGGER.info(f"Returning the sales predictions.")
            return sales_preds
        except (ValueError, KeyError, TypeError) as e:
            if "not aligned" in str(e):
                LOGGER.error(
                    f"Shape mismatch error during prediction. Check input data shape and model weights. {e}"
                )
            else:
                LOGGER.error(f"Failed to make predictions: {e}")
            return None


def output_intermediate_file(
    data: Union[pd.DataFrame, SalesPredictions, SalesAttributes],
    base_filename: str,
    file_type: str = "csv",
) -> None:
    """
    Saves output data to a file (CSV or JSON) with a timestamped filename in the specified folder.

    Args:
        data (Union[pd.DataFrame, SalesPredictions]): The DataFrame or Pydantic model to save.
        base_filename (str): The base filename to use (without extension).
        file_type (str): The type of the file to save ('csv' or 'json'). Defaults to 'csv'.
    """
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{base_filename}_{timestamp}.{file_type}"
        file_path = Path(SETTINGS.INTERMEDIATE_OUTPUT_PATH) / filename

        if isinstance(data, pd.DataFrame) and file_type == "csv":
            data.to_csv(file_path)
        elif (
            isinstance(data, (SalesPredictions, SalesAttributes))
            and file_type == "json"
        ):
            with open(file_path, "w") as f:
                f.write(data.json())

        LOGGER.info(f"Saved intermediate data to: {file_path}")
    except (TypeError, PermissionError, FileNotFoundError, IOError) as e:
        LOGGER.error(f"Failed to save intermediate data: {e}")


def load_model(model_path: str):
    """
    This function loads the trained model from a pickle file.

    Args:
        model_path (str): The path of the pickle file.

    Returns:
        The trained model.
    """

    # Validate the path of the pickle file
    model_path = Path(model_path)
    if not model_path.exists():
        LOGGER.error(f"No model found at {model_path}")
        return None

    # Open the pickle file and load the model
    with open(model_path, "rb") as file:
        try:
            model = pickle.load(file)
        except (pickle.UnpicklingError, EOFError, pickle.PickleError) as e:
            LOGGER.error(f"Failed to load model from {model_path}: {e}")
            return None

    return model


# Loading the predictive model during the initialization of the module
model = load_model(SETTINGS.PRED_MODEL_PATH)
if model is not None:
    try:
        PRED_MODEL = PredictionModel(model, SETTINGS.SALES_CLASS_NAMES)
    except (TypeError, ValueError) as e:
        LOGGER.error(
            f"Error occurred while initializing the prediction model: {str(e)}"
        )
        PRED_MODEL = None
else:
    LOGGER.error("Failed to load the prediction model.")
    PRED_MODEL = None
