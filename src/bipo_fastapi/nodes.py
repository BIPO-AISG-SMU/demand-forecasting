import pandas as pd
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
CONF_LOADER = ConfigLoader(conf_source=settings.CONF_SOURCE)
CONF_PARAMS = CONF_LOADER["parameters"]
CONF_CONST = CONF_LOADER.get("constants*")
CONF_INFERENCE = CONF_LOADER.get("inference*")


def transform_mkt_input(mkt_df: pd.DataFrame) -> pd.DataFrame:
    """Function to prepare marketing dataframe for transform_model_input function.
    This function will rename and convert date related string to datetime format, and convert the campaign name to string.

    Args:
        mkt_df (pd.DataFrame): Raw inference Marketing dataframe.

    Returns:
        mkt_df (pd.DataFrame): Processed inference Marketing dataframe.
    """
    # Rename the columns to the raw column input
    new_column_names = {
        "campaign_name": "Name",
        "campaign_start_date": "Date Start",
        "campaign_end_date": "Date End",
        "marketing_channels": "Mode",
        "total_cost": "Total Cost",
    }
    # Rename the columns
    mkt_df.rename(columns=new_column_names, inplace=True)

    # Set "Date Start" and "Date End" to datetime
    mkt_df["Date End"] = pd.to_datetime(mkt_df["Date End"])
    mkt_df["Date Start"] = pd.to_datetime(mkt_df["Date Start"])
    LOGGER.info("Completed transform marketing input")
    return mkt_df


def impute_lag_sales_df(
    lag_sales_df: pd.DataFrame, outlet_df: pd.DataFrame
) -> pd.DataFrame:
    """Imputes lag sales if the given lag sales is less than the required number of days (ie. 14 days from today when the prediction is made). It is assumed that at least, the first 7 days of lag sales is given, as the most recent data might be unavailable. (E.g today is 18 Oct, expected lag sales is 4-17 Oct, and minimum lag sales to be given from 4-10 Oct).
    - Extends the df by adding empty rows until the last date of the inference period. This ensures there is enough rows to shift forward when generating lag features.
    - Also renames lag sales column to proxyrevenue.
    - Assumes the dates are all 1 day increments. (ie. not 4 may, 6 may..).

    Also, add empty rows to match length of outlet_df. As generation of lag features and tsfresh will shift the values forward to match prediction period.

    Args:
        lag_sales_df (pd.DataFrame): lag sales df.
        outlet_df (pd.DataFrame): Outlet data.

    Raises:
        None.

    Returns:
        pd.DataFrame: Lag sales df with complete values.
    """
    lookback_period = CONF_INFERENCE["lookback_period"]

    # Impute if lag sales is less than lookback period (e.g 14 days)
    if len(lag_sales_df) < lookback_period:
        LOGGER.info(f"Lag sales has less than {lookback_period}")
        today_date = date.today()
        # Check if first 7 days of lag sales are given
        start_date = today_date - timedelta(days=lookback_period)
        if (
            lag_sales_df.loc[start_date : start_date + timedelta(days=6)]
            .notnull()
            .all()
            .all()
        ):
            # Impute by mapping days with missing value to the same day of the previous week and fill up the lookback period
            LOGGER.info("Impute missing lag sales values")
            # Fill up df to end of lookback period
            end_date = lag_sales_df.index[0] + timedelta(days=lookback_period - 1)
            lag_sales_df = fill_date_range(lag_sales_df, end_date)
            lag_sales_df["day"] = lag_sales_df.index.dayofweek
            lag_sales_df["lag_sales"] = lag_sales_df.groupby("day")["lag_sales"].fillna(
                method="ffill"
            )
            lag_sales_df.drop(["day"], axis=1, inplace=True)
            # Rename lag sales to proxyrevenue
            lag_sales_df.rename(
                columns={
                    "lag_sales": f"{CONF_PARAMS['columns_to_create_lag_features'][0]}"
                },
                inplace=True,
            )
            # Fill up df to end of prediction period
            lag_sales_df = fill_date_range(lag_sales_df, outlet_df.index[-1])
            lag_sales_df = lag_sales_df.fillna(0)
            return lag_sales_df
        else:
            LOGGER.error("Missing values in first 7 days of lag sales data")
            return lag_sales_df
    # Lag sales is equal to or more than lookback_period
    else:
        # Add empty rows to match prediction period
        end_date = outlet_df.index[-1]
        lag_sales_df = fill_date_range(lag_sales_df, end_date)
    return lag_sales_df


def fill_date_range(df: pd.DataFrame, end_date: str) -> pd.DataFrame:
    """Generate extra empty rows to fill the given dataframe up to the specified date.

    Args:
        df (pd.DataFrame): Given dataframe.
        end_date (str): End date.

    Returns:
        pd.DataFrame: Dataframe that is filled with empty rows to the specified date.
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


def save_partition_dataset(df: pd.DataFrame, partition_filepath: str) -> pd.DataFrame:
    """Save memorydataset from previous node output as a csv dataset, which is then loaded as a partitioned dataset for downstream nodes which require partitioned dataset as input.

    Args:
        df (pd.DataFrame): Memorydataset
        partition_filepath (str): The file path where the partitioned dataset should be saved.

    Returns:
        pd.DataFrame: The DataFrame loaded as a partitioned dataset.
    """
    # Rename Date column to date
    df.index.names = [CONF_CONST["default_date_col"]]

    # Save as csv file
    output = CSVDataSet(
        filepath=f"{partition_filepath}/{CONF_CONST['inference']['lag_sales_filename']}.csv",
        load_args={"index_col": CONF_CONST["default_date_col"]},
        save_args={
            "index": True,
            "index_label": CONF_CONST["default_date_col"],
            "date_format": "%Y-%m-%d",
        },
    )
    io = DataCatalog(data_sets={CONF_CONST["inference"]["lag_sales_filename"]: output})
    io.save(CONF_CONST["inference"]["lag_sales_filename"], df)
    # Load as incremental dataset
    partition_df = IncrementalDataset(
        path=partition_filepath,
        dataset=CustomCSVDataSet,
        filename_suffix=".csv",
    )
    return partition_df.load()


def convert_to_string(df: pd.DataFrame) -> pd.DataFrame:
    """Convert campaign name column to a string datatype. This is unnecessary in the data pipeline as the dataframe has been saved as a csv file and the campaign name is saved as a string.

    Args:
        df (pd.DataFrame): Input DataFrame with a column to be converted to string.

    Returns:
        pd.DataFrame: Dataframe with transformed campaign name column.
    """
    df[CONF_PARAMS["fe_mkt_column_name"]] = df[
        CONF_PARAMS["fe_mkt_column_name"]
    ].astype(str)
    return df


def process_marketing_df(df: pd.DataFrame, base_df: pd.DataFrame) -> pd.DataFrame:
    """Add Date index name to df generated by create_mkt_campaign_counts_start_end. Merge based on common index with the base_df generated by load_and_structure_marketing_data.

    Args:
        df (pd.DataFrame): Dataframe containing engineered marketing features.
        base_df (pd.DataFrame): Base marketing dataframe.

    Returns:
        pd.DataFrame: Final processed marketing dataframe.
    """
    gen_mkt_df.index.names = [conf_const["default_date_col"]]
    df = base_df.merge(gen_mkt_df, how="left", left_index=True, right_index=True)
    return df

def merge_adstock_and_marketing_df(adstock_df: pd.DataFrame,marketing_df:pd.DataFrame) -> pd.DataFrame:
    """merge adstock features and marketing dataframe 

    Args:
        adstock_df (pd.DataFrame): adstock features dataframe
        marketing_df (pd.DataFrame): marketing dataframe
    Returns:
        pd.DataFrame: final processed marketing dataframe with adstock features
    """
    adstock_df.index.names = [conf_const["default_date_col"]]
    df = adstock_df.merge(marketing_df, how="left", left_index=True, right_index=True)
    return df

def merge_generated_features_to_main(
    generated_features_df: pd.DataFrame, main_df: pd.DataFrame
) -> pd.DataFrame:
    """Merge generated_features_df to the main_df for marketing and lag_sales pipeline.

    Args:
        generated_features_df (pd.DataFrame): Generated_features_df.
        main_df (pd.DataFrame): Merged df consisting of lag_sales, outlet, marketing.

    Returns:
        pd.DataFrame: DataFrame resulting from joining main_df and generated_features_df.
    """
    # lag_sales df
    if isinstance(generated_features_df, dict):
        key = list(generated_features_df.keys())[0]
        generated_features_df = generated_features_df[key]
    if isinstance(main_df, dict):
        key = list(main_df.keys())[0]
        main_df = main_df[key]
    return main_df.join(generated_features_df)


def merge_pipeline_output(
    lag_sales_df: dict, marketing_df: pd.DataFrame, outlet_df: pd.DataFrame
) -> pd.DataFrame:
    """Merge outputs from lag_sales, marketing, and outlet pipelines. lag_sales.

    Args:
        lag_sales_dict (dict): Output from lag_sales pipeline.
        marketing_df (pd.DataFrame): Output from marketing pipeline.
        outlet_df (pd.DataFrame): Output from outlet pipeline.

    Returns:
        pd.DataFrame: Merged dataframe.
    """
    if isinstance(lag_sales_df, dict):
        key = list(lag_sales_df.keys())[0]
        lag_sales_df = lag_sales_df[key]
    merged_df = lag_sales_df.join(marketing_df)
    merged_df = merged_df.join(outlet_df)
    adstock_columns = merged_df.filter(like="adstock").columns
    merged_df[adstock_columns] = merged_df[adstock_columns].fillna(0)
    return merged_df


def process_final_merged_df(merged_df: pd.DataFrame) -> pd.DataFrame:
    """Drop unnecessary columns and convert boolean features to numeric.

    Args:
        merged_df (pd.DataFrame): Final merged dataframe after processing and feature engineering.

    Returns:
        merged_df (pd.DataFrame): The processed DataFrame with unnecessary columns removed.
                      and boolean features converted to numeric.
    """
    merged_df.drop(CONF_INFERENCE["columns_to_drop_list"], axis=1, inplace=True)
    # Convert bool features to numeric as orderedmodel cannot take in bool features
    for col in CONF_INFERENCE["bool_to_numeric_col_list"]:
        try:
            merged_df[col] = pd.to_numeric(merged_df[col], errors="coerce").astype(int)
        except:
            LOGGER.info(f"{col} is not found in merged df")
    return merged_df
