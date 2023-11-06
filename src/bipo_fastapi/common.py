from typing import List
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from interpret.glassbox import ExplainableBoostingClassifier
from kedro.config import ConfigLoader
from bipo import settings
import logging

LOGGER = logging.getLogger(settings.LOGGER_NAME)
CONF_LOADER = ConfigLoader(conf_source=settings.CONF_SOURCE)


def generate_default_daily_date(num_dates: int = 7) -> List:
    """Function to generate default dates for inference.

    Args:
        num_dates (int, optional): Number of days. Defaults to 7 days.

    Returns:
        num_days_list (List): list of dates for inference in the format 'YYYY-MM-DD'.
    """
    # Generate inference list
    today_date = datetime.today()

    # Create a list to store the dates for the next 7 days
    num_days_list = []
    
    # Loop to generate dates for the next 7 days
    days = 2  # inference period is after 2 days
    for day in range(num_dates):
        day = days + day
        next_day = today_date + timedelta(days=day)
        num_days_list.append(next_day.strftime("%Y-%m-%d"))
    return num_days_list


def generate_default_lag_date(num_dates: int = 14) -> List:
    """Function to generate default lag dates for inference.

    Args:
        num_dates (int, optional): Number of days. Defaults to 14.

    Returns:
        previous_days_list (List): list of lag dates for inference.
    """
    # Generate inference list
    today_date = datetime.today()
    # Create a list to store the dates for the next 7 days
    previous_days_list = []
    # Loop to generate dates for the next 7 days
    day_difference = 14
    for day in range(num_dates):
        next_day = today_date - timedelta(days=day_difference)
        previous_days_list.append(next_day.strftime("%Y-%m-%d"))
        day_difference -= 1
    return previous_days_list


def explain_ebm_inference(
    model: ExplainableBoostingClassifier,
    model_input_df: pd.DataFrame,
    pred_y_list: list,
    date_list: pd.Series
):
    """EBM explainability for each single prediction. This function is used in inference pipeline.

    1) local: breakdown of how much each term contributed to the prediction for a single sample. For inference. The intercept reflects the average case. In regression, the intercept is the average y-value of the train set (e.g., $5.51 if predicting cost). In classification, the intercept is the log of the base rate (e.g., -2.3 if the base rate is 10%).

    Args:
        model (ExplainableBoostingClassifier): EBM model weights.
        partitioned_input (Dict[str, pd.DataFrame]):  Kedro IncrementalDataSet Dictionary containing fold-based training dataset containing features and target variables identified with suffixes _X and _y.
        x (pd.Series, optional): 1 row of the X test dataset for local explainability, which corresponds to a single inference point.
        y (float, optional): Single actual y value for local explainability.

    Returns:
        plotly.graph_objs: plotly figure. Also saves the plotly figure/static image locally.
    """
    conf_params = CONF_LOADER["parameters"]
    conf_const = CONF_LOADER.get("constants*")
    output_type = conf_params["output_type"]
    output_filepath = conf_const["inference"]["ebm_explain_filepath"]

    for row in range(len(model_input_df)):
        x = model_input_df.iloc[row]
        y = pred_y_list[row].argmax()
        # Convert y from float to string as predicted is string
        ebm_explanation = model.explain_local(np.array(x), str(y))
        plotly_fig = ebm_explanation.visualize(0)
        if not os.path.exists(output_filepath):
            os.makedirs(output_filepath)
        if output_type == "png":
            filepath = os.path.join(
                output_filepath,
                f"local_contribution_to_prediction_{date_list[row]}.png",
            )
            plotly_fig.write_image(filepath)
        elif output_type == "html":
            filepath = os.path.join(
                output_filepath,
                f"local_contribution_to_prediction_{date_list[row]}.html",
            )
            plotly_fig.write_html(filepath)
        else:
            LOGGER.error("Invalid value for output_type. Accepts either png or html")
    return True