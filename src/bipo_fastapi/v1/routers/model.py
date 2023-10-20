import logging
from fastapi import APIRouter, HTTPException, status, Response

from bipo_fastapi.config import SETTINGS
from bipo_fastapi.deps import PRED_MODEL
from bipo_fastapi.schemas import SalesAttributes, SalesPrediction, SalesPredictions

LOGGER = logging.getLogger("kedro")
router = APIRouter()


@router.get("/version")
def version():
    """
    Get version (UUID) of the model
    - this endpoint returns the version (UUID) of the model currently loaded in the API.

    Returns:
        dict: A dictionary with a single key-value pair. The key is "version" and the value is the model version (UUID).

    """
    LOGGER.info("Received request for model version.")
    return {"Model Version (UUID)": SETTINGS.PRED_MODEL_UUID}


@router.post(
    "/predict", response_model=SalesPredictions, status_code=status.HTTP_201_CREATED
)
def predict_sales(request: SalesAttributes, response: Response):
    """
    Predict sales classes based on the provided attributes.
    - this endpoint receives a POST request containing a list of sales attributes,
    - makes sales class predictions using a trained model, and
    - returns the predictions.

    Args:
        request (SalesAttributes): A list of sales attributes for which to predict sales classes.

    Returns:
        SalesPredictions: A list of predicted sales classes.

    Raises:
        HTTPException: If the request data is invalid.
    """

    # Validate the sales request data
    LOGGER.info("API request received.")

    if (
        not request.outlet_attributes
        and not request.lag_sales_attributes
        and not request.mkt_attributes
    ):
        error_msg = "Invalid API request data provided."
        LOGGER.error(error_msg)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=error_msg)

    # Make predictions using the PredictionModel instance
    LOGGER.info("Initiating model inference.")
    sales_predictions = PRED_MODEL.predict(request)

    if sales_predictions is not None:
        # Log the successful prediction
        LOGGER.info("Model inference completed successfully.")

        # Return the SalesPredictions instance
        return sales_predictions
    else:
        # Handling invalid prediction
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        error_msg = "Model inference failed: No predictions returned."
        LOGGER.error(error_msg)

        return SalesPredictions(
            sales_predictions=[
                SalesPrediction(
                    date="",
                    cost_centre_code=0,
                    sales_class_id="",
                    sales_class_name="",
                    probabilities=[],
                )
            ]
        )
