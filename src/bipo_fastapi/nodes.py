import pandas as pd
import numpy as np
from datetime import date, timedelta
from typing import Dict
# Create a logger for this module
from bipo_fastapi.custom_dataset import CustomCSVDataSet
from kedro_datasets.pandas import CSVDataSet
from kedro.io import DataCatalog
from kedro.io import IncrementalDataset
from kedro.config import ConfigLoader
from bipo import settings
import logging

LOGGER = logging.getLogger(settings.LOGGER_NAME)
conf_loader = ConfigLoader(conf_source=settings.CONF_SOURCE)
conf_params = conf_loader["parameters"]
conf_const = conf_loader.get("constants*")
conf_inference = conf_loader.get("inference*")

def transform_mkt_input(mkt_df: pd.DataFrame)->pd.DataFrame:
    """Function to prepare marketing dataframe for transform_model_input function.
    This function will rename and convert date related string to datetime format, and convert the campaign name to string

    Args:
        mkt_df (pd.DataFrame): Raw inference Marketing dataframe

    Returns:
        mkt_df (pd.DataFrame): Processed inference Marketing dataframe
    """
    # rename the columns to the raw column input
    new_column_names = {
        "campaign_name": "Name",
        "campaign_start_date": "Date Start",
        "campaign_end_date": "Date End",
        "marketing_channels": "Mode",
        "total_cost": "Total Cost",
    }
    # Rename the columns
    mkt_df.rename(columns=new_column_names, inplace=True)

    # set "Date Start" and "Date End" to datetime
    mkt_df["Date End"] = pd.to_datetime(mkt_df["Date End"])
    mkt_df["Date Start"] = pd.to_datetime(mkt_df["Date Start"])
    LOGGER.info("Completed transform marketing input")
    return mkt_df


def impute_lag_sales_df(
    lag_sales_df: Dict[str, str], outlet_df: pd.DataFrame
) -> pd.DataFrame:
    """
    - Imputes lag sales if the given lag sales is less than the required number of days (ie. 14 days from today when the prediction is made). It is assumed that at least, the first 7 days of lag sales is given, as the most recent data might be unavailable. (E.g today is 18 Oct, expected lag sales is 4-17 Oct, and minimum lag sales to be given from 4-10 Oct).
    - Extends the df by adding empty rows until the last date of the inference period. This ensures there is enough rows to shift forward when generating lag features. 
    - Also renames lag sales column to proxyrevenue. 
    - Assumes the dates are all 1 day increments. (ie. not 4 may, 6 may..)

    Also, add empty rows to match length of outlet_df. As generation of lag features and tsfresh will shift the values forward to match prediction period.

    Args:
        lag_sales_df (pd.DataFrame): lag sales df
        outlet_df (pd.DataFrame): outlet df

    Raises:
        None

    Returns:
        pd.DataFrame: Lag sales df with complete values
    """    
    # impute if lag sales is less than lookback period (e.g 14 days)
    if len(lag_sales_df) < conf_params["lookback_period"]:
        LOGGER.info(f"Lag sales has less than {conf_params['lookback_period']}")
        today_date = date.today()
        # check if first 7 days of lag sales are given
        start_date = today_date - timedelta(days=conf_params["lookback_period"])
        if (
            lag_sales_df.loc[start_date : start_date + timedelta(days=6)]
            .notnull()
            .all()
            .all()
        ):
            # impute by mapping days with missing value to the same day of the previous week and fill up the lookback period
            LOGGER.info("Impute missing lag sales values")
            # fill up df to end of lookback period
            end_date = lag_sales_df.index[0] + timedelta(days=conf_params["lookback_period"] - 1)
            lag_sales_df = fill_date_range(lag_sales_df, end_date)
            lag_sales_df["day"] = lag_sales_df.index.dayofweek
            lag_sales_df["lag_sales"] = lag_sales_df.groupby("day")["lag_sales"].fillna(
                method="ffill"
            )
            lag_sales_df.drop(["day"],axis=1,inplace=True) 
            # rename lag sales to proxyrevenue
            lag_sales_df.rename(columns={"lag_sales": f"{conf_params['columns_to_create_lag_features'][0]}"},inplace=True)
            # fill up df to end of prediction period
            lag_sales_df = fill_date_range(lag_sales_df, outlet_df.index[-1])
            lag_sales_df = lag_sales_df.fillna(0)
            return lag_sales_df
        else:
            LOGGER.error("Missing values in first 7 days of lag sales data")
            return lag_sales_df
    # lag sales is equal to or more than lookback_period
    else:
        # add empty rows to match prediction period
        end_date = outlet_df.index[-1]
        lag_sales_df = fill_date_range(lag_sales_df, end_date)
    return lag_sales_df


def fill_date_range(df: pd.DataFrame, end_date: str)->pd.DataFrame:
    """generate extra empty rows to fill the given dataframe up to the specified date

    Args:
        df (pd.DataFrame): given dataframe
        end_date (str): end date

    Returns:
        pd.DataFrame: dataframe that is filled with empty rows to the specified date
    """
    if end_date <= df.index[-1]:
        LOGGER.info(
            "Unable to fill dateframe to given end date as end date is within given dataframe"
        )
        return df
    start_date = df.index[-1] + timedelta(days=1)
    empty_df = pd.DataFrame(
        index=pd.date_range(
            start=start_date,
            end=end_date,
            freq="D",
        )
    )
    df = pd.concat([df, empty_df])
    return df


def save_partition_dataset(df: pd.DataFrame, partition_filepath: str)->pd.DataFrame:
    """Save memorydataset from previous node output as a csv dataset, which is then loaded as a partitioned dataset for downstream nodes which require partitioned dataset as input.

    Args:
        df (pd.DataFrame): memorydataset
        partition_filepath (str): 

    Returns:
        bool: True. This function only returns True to faciliate unit test. The main purpose of this function is to save the memorydataset locally as a csv file.
    """
    # rename Date column to date
    df.index.names = [conf_const["default_date_col"]]
    # base_filename = partition_filepath.split("10_model_inference_output/")[-1]

    # save as csv file
    output = CSVDataSet(
        filepath=f"{partition_filepath}/{conf_const['inference']['lag_sales_filename']}.csv",
        load_args={"index_col": conf_const["default_date_col"]},
        save_args={"index": True, "index_label": conf_const["default_date_col"], "date_format": "%Y-%m-%d"},
    )
    io = DataCatalog(data_sets={conf_const["inference"]['lag_sales_filename']: output})
    io.save(conf_const["inference"]['lag_sales_filename'], df)
    # load as incremental dataset
    partition_df = IncrementalDataset(
        path=partition_filepath,
        dataset=CustomCSVDataSet,
        filename_suffix=".csv",
    )
    return partition_df.load()

def convert_to_string(df:pd.DataFrame)->pd.DataFrame:
    """convert campaign name column to a string datatype. This is unnecessary in the data pipeline as the dataframe has been saved as a csv file and the campaign name is saved as a string. 

    Args:
        df (pd.DataFrame): 

    Returns:
        pd.DataFrame: dataframe with transformed campaign name column
    """
    df[conf_params["fe_mkt_column_name"]] = df[conf_params["fe_mkt_column_name"]].astype(str)
    return df 

def process_marketing_df(df:pd.DataFrame,base_df:pd.DataFrame)->pd.DataFrame:
    """
    - add Date index name to df generated by create_mkt_campaign_counts_start_end
    - merge based on common index with the base_df generated by load_and_structure_marketing_data.

    Args:
        df (pd.DataFrame): dataframe containing engineered marketing features
        base_df (pd.DataFrame): base marketing dataframe

    Returns:
        pd.DataFrame: final processed marketing dataframe
    """
    df.index.names = [conf_const["default_date_col"]]
    return df.join(base_df) 

def merge_generated_features_to_main(generated_features_df:pd.DataFrame,main_df:pd.DataFrame)->pd.DataFrame:
    """merge generated_features_df to the main_df for marketing and lag_sales pipeline

    Args:
        generated_features_df (pd.DataFrame): generated_features_df
        main_df (pd.DataFrame): merged df consisting of lag_sales, outlet, marketing 
    """
    # lag_sales df 
    if isinstance(generated_features_df,dict):
        key = list(generated_features_df.keys())[0]
        generated_features_df = generated_features_df[key]
    if isinstance(main_df,dict):
        key = list(main_df.keys())[0]
        main_df = main_df[key]
    return main_df.join(generated_features_df)



def merge_pipeline_output(lag_sales_df:dict, marketing_df:pd.DataFrame, outlet_df:pd.DataFrame)->pd.DataFrame:
    """merge outputs from lag_sales, marketing, and outlet pipelines. lag_sales  

    Args:
        lag_sales_dict (dict): output from lag_sales pipeline
        marketing_df (pd.DataFrame): output from marketing pipeline
        outlet_df (pd.DataFrame): output from outlet pipeline

    Returns:
        pd.DataFrame: merged dataframe
    """
    if isinstance(lag_sales_df,dict):
        key = list(lag_sales_df.keys())[0]
        lag_sales_df = lag_sales_df[key]
    merged_df = lag_sales_df.join(marketing_df)
    merged_df = merged_df.join(outlet_df)
    return merged_df

def process_final_merged_df(merged_df:pd.DataFrame)->pd.DataFrame:
    """
    - drop unnecessary columns
    - rename column propensity_factor to factor 
    - convert boolean features to numeric 

    Args:
        merged_df (pd.DataFrame): final merged dataframe after processing and feature engineering
    """
    merged_df.drop(conf_inference["columns_to_drop_list"],axis=1,inplace=True)
    merged_df.rename(columns=conf_inference["inference_columns_rename_map"],inplace=True)
    # convert bool features to numeric as orderedmodel cannot take in bool features
    for col in conf_inference["bool_to_numeric_col_list"]:
        merged_df[col] = pd.to_numeric(
                merged_df[col], errors="coerce"
            ).astype(int)
    return merged_df
