import pandas as pd
import os
import logging
import warnings
from datetime import datetime, date
import sys

from .data_check import DataCheck
from kedro.config import ConfigLoader
from kedro.io import DataCatalog
from kedro.framework.project import settings
from kedro_datasets.pandas import (
    CSVDataSet,
    ExcelDataSet,
)

from bipo.utils import (
    get_project_path,
    create_dir_from_project_root,
)  # get_logger,

# from bipo import utils

# initiate to constants.yml and parameters.yml
project_path = get_project_path()
conf_loader = ConfigLoader(conf_source=project_path / "conf")
conf_params = conf_loader["parameters"]["dataloader"]
constants = conf_loader.get("constants*")["dataloader"]

# logging = logging.getLogger("kedro")
logging = logging.getLogger(__name__)

# FOR RENAMING COLUMNS
NEW_COLUMN_NAMES = constants["new_column_names"]

# ADJUSTABLE PARAMETERS
DATE_RANGE = conf_params["date_range"]
TARGET_VARIABLE = conf_params["target_feature"]
ZERO_PERCENTAGE_THREHOLD = conf_params["zero_percentage_threshold"]

# add_save_filepath_to_catalog from dataloader.py will populate the filepath to save in processed_dataset_catalog.
processed_dataset_catalog = DataCatalog()


def check_outlet_location(proxy_revenue_df: pd.DataFrame, cost_centre_code: str):
    """Checks the location of the outlet.
    Args:
        proxy_revenue_df: input from proxy revenue data
        cost_centre_code: individual cost centre code
    Returns:
        None
    """
    mapping = dict(
        zip(proxy_revenue_df["CostCentreCode"], proxy_revenue_df["Location"])
    )
    outlet_region_mapping_df = pd.DataFrame(
        list(mapping.items()), columns=["CostCentreCode", "Location"]
    )

    location = outlet_region_mapping_df.loc[
        outlet_region_mapping_df["CostCentreCode"] == cost_centre_code, "Location"
    ].values[0]
    return location


def check_number_of_zeros_proxy_revenue(
    proxy_revenue_df: pd.DataFrame, cost_centre_code: str
):
    """Function to check for percentage of zeros in proxy_revenue. It will give a error message if the percentage of zeros is higher than the given threshold. Can be seen under info.log as a ERROR.


    Args:
        proxy_revenue_df (pd.DataFrame): input from proxy revenue data
        cost_centre_code (str): individual cost centre code

    Returns:
        cost_centre_code (str): Cost Centre Code that has more percentage of zeros than threshold.
    """
    # Filter proxy_revenue_df based on cost centre
    proxy_revenue_df = proxy_revenue_df[
        proxy_revenue_df["CostCentreCode"] == cost_centre_code
    ]

    # Get the percentage of zeros in proxy revenue
    zero_percentage_threshold = (
        ZERO_PERCENTAGE_THREHOLD  # parameters["zero_percentage_threshold"]
    )
    target_variable = TARGET_VARIABLE  # parameters["target_variable"]

    zero_percentage = (
        proxy_revenue_df[target_variable].eq(0).sum()
        / len(proxy_revenue_df[target_variable])
        * 100
    )

    # if percentage of zeros in proxy revenue is more than 10%. Give a warning to exclude the cost centre from pipeline.
    if zero_percentage > zero_percentage_threshold:
        logging.info(
            f"Cost Centre Code '{cost_centre_code}' has {zero_percentage:.2f}% missing values, which is more than threshold {zero_percentage_threshold}%. Excluding Cost Centre Code '{cost_centre_code}' from pipeline"
        )
        return cost_centre_code


def add_save_filepath_to_catalog(proxy_revenue_df: pd.DataFrame, cost_centre_code: str):
    """Function to append the cost centre data catalog infomation into processed_dataset_catalog.

    Add location infomation to specific outlet to processed_transaction_dataset_catalog.

    Args:
        proxy_revenue_df (pd.DataFrame): input from proxy revenue data
        code (str): individual cost centre code

    Returns:
        None
    """
    location = check_outlet_location(
        proxy_revenue_df,
        cost_centre_code,
    )
    if cost_centre_code not in processed_dataset_catalog.list():
        # Create a new dataset and add it to the catalog
        dataset = CSVDataSet(
            filepath=f"data/02_data_loading/merged_{cost_centre_code}_{location}.csv"
        )
        processed_dataset_catalog.add(cost_centre_code, dataset)
    return None


def save_data(merged_df: pd.DataFrame, cost_centre_code: str):
    """Function to save the merged revenue dataset as a csv file in 02_data_loading

    Args:
        cost_centre_code (str): Cost centre code. Eg 201, 202, etc
        merged_df (pandas.core.frame.DataFrame): merged transaction dataframe

    Returns:
        None
    """
    processed_dataset_catalog.save(cost_centre_code, merged_df)
    return None


def merge_all_data(
    preprocessed_proxy_revenue_data,
    preprocessed_propensity_data,
    preprocessed_climate_data,
    preprocessed_covid_data,
    preprocessed_holiday_data,
    preprocessed_marketing_data,
) -> pd.DataFrame:
    """Merge all datasets into a single dataframe for the specified outlet with the help of each dataset class.
    Checks if the merged dataframe has the expected number of columns, filters for selected date range, and saves it in the specified filepath.

    Args:
        preprocessed_proxy_revenue_data (pandas.core.frame.DataFrame): Preprocessed Transaction dataframe
        preprocessed_propensity_data (pandas.core.frame.DataFrame): Preprocessed Consumer Propensity dataframe
        preprocessed_climate_data (pandas.core.frame.DataFrame: Preprocessed Climate dataframe
        preprocessed_covid_data (pandas.core.frame.DataFrame): Preprocessed covid dataframe
        preprocessed_holiday_data (pandas.core.frame.DataFrame): Preprocessed holiday dataframe
        preprocessed_marketing_data (pandas.core.frame.DataFrame): Preprocessed marketing dataframe

    Returns:
        merged_outlet_region.csv (pandas.core.frame.DataFrame): merged csv file saved into the specified filepath
    """
    datacheck = DataCheck()
    combine_df = pd.concat(
        [
            preprocessed_proxy_revenue_data,
            preprocessed_propensity_data,
            preprocessed_climate_data,
            preprocessed_covid_data,
            preprocessed_holiday_data,
            preprocessed_marketing_data,
        ],
        axis=1,
    )
    # check dataframe is not empty and columns tally
    datacheck.check_data_exists(combine_df)
    combine_df_cols = combine_df.columns.tolist()
    datacheck.check_target_feature(combine_df_cols)
    datacheck.check_columns(combine_df_cols, "merged_df")

    # Filter selected date range
    combine_df = combine_df.loc[DATE_RANGE[0] : DATE_RANGE[1]]

    # Convert datetime index to date column
    combine_df.reset_index(inplace=True)
    combine_df.rename(columns={"index": "date"}, inplace=True)

    return combine_df
