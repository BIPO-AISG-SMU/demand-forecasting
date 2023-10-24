import pandas as pd
from pathlib import Path
from datetime import datetime
import pickle
from typing import List, Union
from kedro.io import PartitionedDataset
from kedro_datasets.pandas import CSVDataSet
from bipo import settings
from kedro.config import ConfigLoader
from bipo_fastapi.hooks import hook_manager
# from bipo_fastapi.load_data_catalog import catalog
from bipo_fastapi.load_data_catalog import load_data_catalog
# Create a logger for this module
from bipo_fastapi.config import SETTINGS
import logging
conf_loader = ConfigLoader(conf_source=settings.CONF_SOURCE)
conf_const = conf_loader.get("constants*")
conf_inference = conf_loader.get("inference*")
LOGGER = logging.getLogger(settings.LOGGER_NAME)

from bipo_fastapi.schemas import SalesAttributes, SalesPredictions, SalesPrediction
from bipo_fastapi.common import explain_ebm_inference

# from bipo.inference_pipeline.inference_pipeline import run_pipeline
# import pipeline object from kedro_pipeline_test
from bipo_fastapi.pipeline import main_pipeline
from kedro.runner.sequential_runner import SequentialRunner


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

    def transform_attributes(self, sales_attrs: SalesAttributes) -> pd.DataFrame:
        """
        Converts outlet_attributes, mkt_attributes and lag_sales_attributes into DataFrames.
        Note that mkt_attributes can accept mulitple campaign inputs.

        Args:
            sales_attrs (SalesAttributes) : An instance of the SalesAttributes class.

        Returns:
            DataFrames representation of dates_cost_centre and sales_attrs.
        """
        # try:
        outlet_data = [attr.dict() for attr in sales_attrs.outlet_attributes]
        lag_sales_data = [attr.dict() for attr in sales_attrs.lag_sales_attributes]
        mkt_data = [attr.dict() for attr in sales_attrs.mkt_attributes]

        # checks if the length of all the given input has the same length as the date input.
        LOGGER.info(
            f"Checking if the length of all the given input has the same length as the date input."
        )
        # get the date length from the dict and use as reference.
        reference_inference_length = len(list(outlet_data[0].values())[0])
        reference_lag_length = len(list(lag_sales_data[0].values())[0])
        for key, lst in outlet_data[0].items():
            if isinstance(lst, list) and len(lst) != reference_inference_length:
                LOGGER.error(
                    f"{key} has {len(lst)} inputs. Fields must have the same length as {list(outlet_data[0].keys())[0]}: {reference_inference_length}"
                )
        for key, lst in lag_sales_data[0].items():
            if isinstance(lst, list) and len(lst) != reference_lag_length:
                LOGGER.error(
                    f"{key} has {len(lst)} inputs. Fields must have the same length as {list(lag_sales_data[0].keys())[0]}: {reference_lag_length}"
                )
        # convert the request inputs into dataframe
        try:
            # sales_data and lag_sales_data should only have 1 input. If more input is given, error will be raise
            if len(outlet_data) == 1 and len(lag_sales_data) == 1:
                LOGGER.info(
                    f"Converting outlet_attributes and lag_sales_attributes request data into DataFrame"
                )
                outlet_df = pd.DataFrame(outlet_data[0])
                lag_sales_df = pd.DataFrame(lag_sales_data[0])
        except ValueError as e:
            raise ValueError(
                f"Sales_attributes have {len(outlet_data)} request inputs and lag_sales_attributes have {len(lag_sales_data)} request inputs. Both data should only have 1 request input."
            )

        # Only mkt_date can have mulitple keys / dataframes
        # If the mkt request field is left empty, by default a empty mkt df will be generated.
        if len(mkt_data) >= 1:
            LOGGER.info(f"Mkt_attributes have {len(mkt_data)} request inputs")
            mkt_df_list = []
            for num in range(len(mkt_data)):
                mkt_df = pd.DataFrame(mkt_data[num])
                mkt_df_list.append(mkt_df)

            # Concat the mkt df by rows
            LOGGER.info(f"Converting Mkt_attributes request data into DataFrame")
            mkt_df = pd.concat(mkt_df_list, axis=0, join="outer", ignore_index=False)
        else:
            LOGGER.info(
                f"No ongoing marketing camapign. Creating a mock mkt dataframe for model input"
            )
            columns = {
                "campaign_name": str,
                "campaign_start_date": str,
                "campaign_end_date": str,
                "marketing_channels": list,
                "campaign_total_costs": list,
            }
            mkt_df = pd.DataFrame(columns=columns)

        # save intermediate csv files 
        LOGGER.info(f"Saving model input data.")
        output_intermediate_file(outlet_df, conf_const["inference"]["outlet_filename"])
        output_intermediate_file(lag_sales_df, conf_const["inference"]["lag_sales_filename"])
        output_intermediate_file(mkt_df, conf_const["inference"]["marketing_filename"])

        # get date for inference (use model_input_df)
        dates = outlet_df. iloc[:,0]
        # get cost_centre_codes for labeling
        cost_centre_codes = outlet_df.pop("cost_centre_code")
        return dates, cost_centre_codes

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
            dates, cost_centre_codes = self.transform_attributes(
                sales_attrs
            )
            # load data catalog after saving request as csv files
            catalog = load_data_catalog()
            # load model_input_data from kedro run pipeline
            runner = SequentialRunner()
            output_dict = runner.run(
            pipeline=main_pipeline, catalog=catalog, hook_manager=hook_manager
            )
            # use the final node output in the pipeline as output_dict key
            model_input_data = output_dict["processed_final_merged_df"]
            LOGGER.info(f"Executing model inference.")
            
            # ebm
            if "ebm" in SETTINGS.PRED_MODEL_PATH:
                predictions = self.model.predict_proba(
                model_input_data
                )
                # ebm model explainability
                if conf_inference["enable_explainability"]:
                    explain_ebm_inference(model=self.model,model_input_df =model_input_data,pred_y_list=predictions,date_list=dates)
            # orderedmodel
            elif "ordered" in SETTINGS.PRED_MODEL_PATH:
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
        if isinstance(data, pd.DataFrame) and file_type == "csv":
            # save as partitioned dataset
            filename = f"{base_filename}.{file_type}"
            file_path = Path(SETTINGS.INTERMEDIATE_OUTPUT_PATH)/base_filename
            LOGGER.info(f"{file_path}")
            file_path_csv = str(file_path)
            data_dict = {filename: data}
            data_set = PartitionedDataset(
                path=file_path_csv,
                dataset=CSVDataSet,
            )
            data_set.save(data_dict)
        elif (
            isinstance(data, (SalesPredictions, SalesAttributes))
            and file_type == "json"
        ):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{base_filename}_{timestamp}.{file_type}"
            file_path = Path(SETTINGS.INTERMEDIATE_OUTPUT_PATH) / filename
            LOGGER.info(f"{file_path}")
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