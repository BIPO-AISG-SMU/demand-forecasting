# general imports
import logging
import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
import os
import sys
from bipo.utils import get_project_path
from kedro.config import ConfigLoader

# get filepath to data_loader
current_directory = os.path.dirname(os.path.abspath(__file__))  # current path
parent_directory = os.path.dirname(current_directory)  # bipo
data_loader_path = os.path.join(parent_directory, "pipelines/data_loader/")
data_preprocessing_path = os.path.join(
    parent_directory, "pipelines/data_preprocessing/"
)

sys.path.append(data_loader_path)
sys.path.append(data_preprocessing_path)

# import functions here
from data_check import DataCheck
from read_marketing import ReadMarketingData
from preprocessing_node import DataPreprocessing

# logging = logging.getLogger("kedro")
logging = logging.getLogger(__name__)

# initiate to constants.yml and parameters.yml
project_path = get_project_path()
conf_loader = ConfigLoader(conf_source=project_path / "conf")
conf_params = conf_loader["parameters"]["inference"]
constants = conf_loader.get("constants*")["inference"]

INFERENCE_MARKETING_COLUMNS = constants["inference_marketing_columns"]
INFERENCE_LAG_SALES_COLUMNS = constants["inference_lag_sales_columns"]
INFERENCE_LOCATION_COLUMNS = constants["inference_location_columns"]
INFERENCE_LAG_SALES_DATE_RANGE = conf_params["lag_sales_date_range"]


# initate the classes
preprocess_marketing = ReadMarketingData()
general_preprocessing = DataPreprocessing()


# Class dataloader
class PreprocessInferenceData(DataCheck):  # DataCheck
    """Data Preprocessing for inference pipeline. Inherits the DataCheck class. Accepts 1 input dataframe and output 2 preprocessed dataframe, merged_df and preprocessed_lagged_sales_df.

    Args:
        DataCheck (class): Inherits DataCheck parent class.

    """

    def __init__(self, df: pd.DataFrame):
        super().__init__()
        self.df = df

    def run_pipeline(self) -> pd.DataFrame:
        """Runs a series of preprocessing steps on the input DataFrame.

        Args:
            df (pd.DataFrame): Input DataFrame to be preprocessed.

        Returns:
            preprocessed_location_df, preprocessed_marketing_df,preprocessed_lagged_sales_df (pd.DataFrame): 3 separated Processed DataFrame.
        """
        # Check and Split data into marketing, location and lag_sales
        logging.info("Starting data split for preprocessing")
        self.read_data()
        logging.info("Completed data split")

        # Preprocess marketing data
        # Check if there is data inside
        logging.info("Start preprocessing marketing data")
        self.check_preprocess_mkt_dataframe()
        logging.info("Completed preprocessing marketing data")

        # Preprocess lag_sales data
        logging.info("Start preprocessing lag_sales")
        self.check_preprocess_lag_sales_dataframe()
        logging.info("Completed preprocessing lag_sales")

        # Preprocess location data
        logging.info("Start preprocessing location data")
        self.endo_feature_engineer()
        logging.info("Completed preprocessing location data")

        # Preprocess all data
        logging.info("Start general preprocessing for all data")
        self.preprocess_data()
        logging.info("Completed general preprocessing for all data")

        # Merge marketing and location df
        logging.info("Start merging marketing and location data")
        merged_df, preprocessed_lagged_sales_df = self.merge_data()
        logging.info("Completed merging. Output: Location_df and lagged_sales_df")
        return (merged_df, preprocessed_lagged_sales_df)

    def read_data(self) -> pd.DataFrame:
        """Read, check for expected columns and data exist. Split the data into location_df, lag_sales_df and marketing_df"

        Args:
            inference_df (pd.DataFrame): Input DataFrame to be preprocessed.

        Returns:
            None
        """
        # To be changed
        # df = pd.read_csv(file_path, index_col="date")

        # Check if data have expected columns
        self.check_columns(self.df.columns.tolist(), "inference")

        # Check if there is data inside
        self.check_data_exists(self.df)

        # set index to datetime index
        self.df.index = pd.to_datetime(self.df.index, dayfirst=True)

        # split the df into 3 parts (constants[marketing_columns])
        logging.info(
            f"Generating marketing_df with columns: {INFERENCE_MARKETING_COLUMNS}"
        )
        self.marketing_df = self.df[INFERENCE_MARKETING_COLUMNS].reset_index(drop=True)
        self.marketing_df.dropna(inplace=True)

        logging.info(
            f"Generating lag_sales_df with columns: {INFERENCE_LAG_SALES_COLUMNS}"
        )

        self.lag_sales_df = self.df[
            INFERENCE_LAG_SALES_COLUMNS
        ]  # (constants[lag_sales])

        logging.info(
            f"Generating location_df with columns: {INFERENCE_LOCATION_COLUMNS}"
        )
        self.location_df = self.df[
            INFERENCE_LOCATION_COLUMNS
        ]  # (constants[location_columns])
        return None

    # import from marketing.py
    def preprocess_marketing(self):
        """Convert to start and end date to datetime index. Total cost will be converted to daily cost. An additional column for "cat_mkt_campaign_start" and "cat_mkt_campaign_end" added.

        Args:
            None
        Return:
            None
        """
        # Create a copy
        self.marketing_df = self.marketing_df.copy()

        # # Check for duplicated rows and drop them
        num_duplicates = self.marketing_df.duplicated().sum()
        if num_duplicates > 0:
            logging.info(f"Found {num_duplicates} duplicated rows. Removing them.")
            self.marketing_df = self.marketing_df.drop_duplicates()
        else:
            logging.info("No duplicated rows found.")

        # Replace any '?' with nan
        self.marketing_df.replace("?", np.nan, inplace=True)

        # Convert 'Start Date' and 'End Date' columns to datetime
        self.marketing_df[INFERENCE_MARKETING_COLUMNS[1]] = pd.to_datetime(
            self.marketing_df[INFERENCE_MARKETING_COLUMNS[1]], dayfirst=True
        )
        self.marketing_df[INFERENCE_MARKETING_COLUMNS[2]] = pd.to_datetime(
            self.marketing_df[INFERENCE_MARKETING_COLUMNS[2]], dayfirst=True
        )
        logging.info(
            f"Generating 'cat_mkt_campaign_start' and 'cat_mkt_campaign_end' columns"
        )
        # Generate "cat_mkt_campaign_start" and "cat_mkt_campaign_end"
        self.generate_campaign_start_end_indication()  # self.campaign_fe_df

        # Create a new dataframe with datetime index ranging from 20/3/2023 to 26/3/2023
        index = pd.date_range(start=self.df.index[0], end=self.df.index[-1], freq="D")
        preprocessed_marketing_df = pd.DataFrame(index=index)

        sheet_name = "marketing data"  # can be any name
        feature_of_interest_name = INFERENCE_MARKETING_COLUMNS[0]  # campaign_name
        start_date_name = INFERENCE_MARKETING_COLUMNS[1]
        end_date_name = INFERENCE_MARKETING_COLUMNS[2]
        cost_name = INFERENCE_MARKETING_COLUMNS[3]
        sheet_name_cost = "campaign_daily_cost"
        logging.info(
            f"Get the campaign daily cost and impute NaN values in Cost with median"
        )
        # Get the daily cost and impute NaN values in Cost with median
        marketing_df = preprocess_marketing.get_and_impute_daily_cost(
            sheet_name,
            self.marketing_df,
            start_date_name,
            end_date_name,
            cost_name,
            feature_of_interest_name,
        )
        logging.info(
            f"Convert daily cost and {feature_of_interest_name} to datetime index"
        )
        # Convert daily cost and feature_of_interest_name to datetime index
        preprocessed_marketing_df = preprocess_marketing.set_daily_cost(
            self.marketing_df,
            preprocessed_marketing_df,
            start_date_name,
            end_date_name,
            feature_of_interest_name,
            sheet_name_cost,
        )

        # rename columns (For initial deployment, we assume that only tv_ads are provided)
        column_mapping = dict(
            zip(
                ["campaign_name", "campaign_daily_cost"],
                ["cat_mkt_campaign", "campaign_daily_cost"],
            )
        )
        self.preprocessed_marketing_df = preprocessed_marketing_df.rename(
            columns=column_mapping
        )

        # merge self.campaign_fe_df and self.preprocessed_marketing_df
        merge_df_list = [self.preprocessed_marketing_df, self.campaign_fe_df]
        self.preprocessed_marketing_df = pd.concat(merge_df_list, axis=1)

        return None

    def generate_campaign_start_end_indication(self):
        """generate "cat_mkt_campaign_start" and "cat_mkt_campaign_end" columns with the values True or False if the campaign start and end date falls on the inference date range

        Returns:
            None
        """
        # create empty df to store campaign engineered features
        self.campaign_fe_df = pd.DataFrame({"date": self.df.index})
        # get campaign start and end dates
        specific_dates_start = set(self.marketing_df[INFERENCE_MARKETING_COLUMNS[1]])
        specific_dates_end = set(self.marketing_df[INFERENCE_MARKETING_COLUMNS[2]])
        # Set markers on start and end dates
        self.campaign_fe_df["cat_mkt_campaign_start"] = self.campaign_fe_df["date"].map(
            lambda x: True if x in specific_dates_start else False
        )
        self.campaign_fe_df["cat_mkt_campaign_end"] = self.campaign_fe_df["date"].map(
            lambda x: True if x in specific_dates_end else False
        )
        self.campaign_fe_df.set_index("date", inplace=True)
        return None

    def preprocess_lag_sales(self):
        """Function convert current lag_sales format into a dataframe map to its respective dates.

        Returns:
            None
        """
        # generate dates. (refactor such that can be configurable)
        start_date = datetime.strptime(
            INFERENCE_LAG_SALES_DATE_RANGE["start_date"], "%Y-%m-%d"
        )
        end_date = datetime.strptime(
            INFERENCE_LAG_SALES_DATE_RANGE["end_date"], "%Y-%m-%d"
        )
        date_list = pd.date_range(start_date, end_date, freq="d")

        # generate sales
        sales_list = self.lag_sales_df.iloc[0].values[0]
        logging.info(f"Sales list of {sales_list} generated")

        # Check if length of date_list is equal to sales_list.
        # The purpose is do a mapping sales_list with date_list
        if len(sales_list) == len(date_list):
            # generate lagged sales dataframe
            self.preprocessed_lagged_sales_df = pd.DataFrame(
                {"date": date_list, "proxy_revenue": sales_list}
            )
            self.preprocessed_lagged_sales_df = (
                self.preprocessed_lagged_sales_df.set_index("date")
            )
        else:
            log_string = "The lengh of date range not the same as lengh of lag sales provided. Please adjust lag_sales_date_range in parameters.yml"
            logging.error(log_string, stack_info=True)
            raise ValueError(log_string)
        return None

    def endo_feature_engineer(self):  # move to endo feature engineer
        """Function to add additional features "cat_covid_group_size_cap" and "cat_day_of_week"

        return
            None
        """
        # move to endo feature engineer
        self.location_df = self.location_df.copy()
        # addition of cat_covid_group_size_cap
        self.location_df["cat_covid_group_size_cap"] = "no limit"

        # addition of cat_day_of_week
        self.location_df["cat_day_of_week"] = self.location_df.index.day_name()

        return None

    # import preprocessing functions here
    def preprocess_data(self):
        """Function to check and preprocess all data imported from DataPreprocessing pipeline.

        return
            None

        """
        self.preprocessed_location_df = general_preprocessing.run_pipeline(
            self.location_df
        )
        self.preprocessed_marketing_df = general_preprocessing.run_pipeline(
            self.preprocessed_marketing_df
        )
        self.preprocessed_lagged_sales_df = general_preprocessing.run_pipeline(
            self.preprocessed_lagged_sales_df
        )
        return None

    def merge_data(self):
        """Function to merge back preprocessed_location_df and preprocessed_marketing_df.

        Args:
            None
        returns:
            self.merged_df
        """
        merge_df_list = [self.preprocessed_location_df, self.preprocessed_marketing_df]
        self.merged_df = pd.concat(merge_df_list, axis=1)
        return self.merged_df, self.preprocessed_lagged_sales_df

    # Check if dataframe is empty and set a default value
    def check_preprocess_mkt_dataframe(self):
        """Function to check marketing_df to see if it is empty. Eg, no ongoing campaign during inference. Create a default empty preprocessed_marketing_df for inference.

        Args:
            None
        returns:
            self.preprocessed_marketing_df
        """
        if self.marketing_df.isna().all().all():
            logging.info(
                "The DataFrame only contains NaN values. Creating a mock marketing_df"
            )
            mock_mkt_columns = [
                "cat_mkt_campaign",
                "campaign_daily_cost",
                "cat_mkt_campaign_start",
                "cat_mkt_campaign_end",
            ]

            self.preprocessed_marketing_df = pd.DataFrame(
                columns=mock_mkt_columns, index=self.location_df.index
            )
            # Fill in default values
            self.preprocessed_marketing_df["campaign_daily_cost"] = 0
            self.preprocessed_marketing_df["cat_mkt_campaign_start"] = False
            self.preprocessed_marketing_df["cat_mkt_campaign_end"] = False
            self.preprocessed_marketing_df["cat_mkt_campaign"] = "No"
            return None
        else:
            self.preprocess_marketing()
            return None

    def check_preprocess_lag_sales_dataframe(self):
        """Function to check lag_sales_df to see if it is empty. Eg, no lag proxy reveune during inference. Create a default empty preprocessed_lagged_sales_df for inference.

        Args:
            None
        returns:
            None
        """
        if all(self.lag_sales_df["lag_sales"].apply(lambda x: len(x) == 0)):
            logging.info(
                "The DataFrame only contains NaN values. Creating a mock lag_sales_df"
            )
            mock_lag_sales_columns = ["proxy_revenue"]

            # generate dates. (refactor such that can be configurable)
            start_date = datetime.strptime(
                INFERENCE_LAG_SALES_DATE_RANGE["start_date"], "%Y-%m-%d"
            )
            end_date = datetime.strptime(
                INFERENCE_LAG_SALES_DATE_RANGE["end_date"], "%Y-%m-%d"
            )
            date_list = pd.date_range(start_date, end_date, freq="d")
            self.preprocessed_lagged_sales_df = pd.DataFrame(
                columns=mock_lag_sales_columns, index=date_list
            )
            # Fill in default values
            self.preprocessed_lagged_sales_df["proxy_revenue"] = 0
            logging.info(
                f"The DataFrame created has a shape: {self.preprocessed_lagged_sales_df.shape}"
            )
            return None
        else:
            self.preprocess_lag_sales()
            return None
