import pandas as pd
from datetime import date, timedelta
import logging
from typing import Union
import sys
import os
from kedro.config import ConfigLoader
from bipo.utils import get_project_path

project_path = get_project_path()
conf_loader = ConfigLoader(conf_source=project_path / "conf")
conf_params = conf_loader["parameters"]
constants = conf_loader.get("constants*")

from bipo.pipelines.feature_engineering.common import get_group_number
from bipo.pipelines.feature_engineering.endo import Endogenous
from bipo.pipelines.feature_engineering.feature_engineering import (
    run_tsfresh_fe_pipeline,
)

LAG_LIST = conf_params["feature_engineering"]["endo"]["lag_days_list"]
WINDOW_LIST = conf_params["feature_engineering"]["endo"]["window_days_list"]
NUM_LIST = conf_params["feature_engineering"]["endo"]["num_weeks_list"]
TARGET_FEATURE = constants["inference"]["inference_proxy_revenue_column"]
CAMPAIGN_SHEET = constants["dataloader"]["campaign_sheet"]
CAMPAIGN_COLUMN = constants["inference"]["campaign_column"]
CAMPAIGN_COUNT_COLUMN = constants["inference"]["campaign_count_column"]

logging = logging.getLogger(__name__)


class FeatureEngineering(Endogenous):
    """Feature engineering for inference pipeline. Inherits the Endogenous class. Accepts 2 dataframes as input after preprocessing.

    Args:
        Endogenous (class): Endogenous feature engineering parent class.
    """

    def __init__(
        self,
        location_df: pd.DataFrame,
        lagged_proxy_revenue_df: pd.DataFrame,
    ):
        """Initializes FeatureEngineering class.

        Args:
            location_df (pd.DataFrame): dataframe containing all features except for lagged proxy revenue.
            lagged_proxy_revenue_df (pd.DataFrame): dataframe containing lagged proxy revenue values.
        """
        super().__init__(location_df)
        self.location_df = location_df
        self.lagged_proxy_revenue_df = lagged_proxy_revenue_df

    def run_pipeline(self) -> Union[None, pd.DataFrame]:
        """Run the entire endogenous feature engineering pipeline.

        Args:
            None

        Raises:
            None

        Returns:
            pd.DataFrame: Dataframe with all feature including newly engineered features.
        """
        logging.info("Starting feature engineering")
        # Construct dataframes representing only transformed variables
        lagged_df = self.generate_lag(
            lag_days_list=LAG_LIST, column_name=TARGET_FEATURE
        )
        sma_df = self.generate_sma_sales(
            window_days_list=WINDOW_LIST, column_name=TARGET_FEATURE
        )
        weekly_average_df = self.generate_lag_avg_weekly_sales(
            num_weeks_list=NUM_LIST,
            sales_column_name=TARGET_FEATURE,
        )
        # generate feature is_weekend
        is_weekend_df = self.convert_to_weekend()

        # Create marketing count and existence related information. Will be None if self.mkt_type_categories_list = None as they depend on it for processing
        active_mkt_counts_df = self.get_campaign_count()
        active_mkt_status_df = self.get_marketing_activity_status(
            marketing_sheet_name=CAMPAIGN_SHEET
        )
        active_mkt_status_df.drop([CAMPAIGN_COLUMN], axis=1, inplace=True)

        try:
            df_merge_list = [
                lagged_df,
                sma_df,
                weekly_average_df,
                active_mkt_counts_df,
                active_mkt_status_df,
                is_weekend_df,
            ]

            # Base copy for use in merging
            concat_df = self.location_df.copy()

            # Merge dataframe, when merging via index, it is important to ensure the index are of same time to avoid merged column values from changing to Nan.
            for df in df_merge_list:
                df.index = pd.to_datetime(df.index)
                concat_df = concat_df.merge(
                    df, how="left", left_index=True, right_index=True
                )

        except ValueError:
            log_string = (
                "Encountered all None when merging. Proceed with None as return"
            )
            logging.error(log_string)
            concat_df = None

        logging.debug(f"Source dataframe contains {self.location_df.shape[1]} features")

        logging.debug(f"Concatenated endogenous dataframe shape: {concat_df.shape}\n")
        logging.info("Completed feature engineering")
        return concat_df

    def generate_lag(self, lag_days_list: list, column_name: str):
        """Creates a dataframe containing generated lag values based on column_name variable representing dataframe column name according to input lag_days_list from a dataframe instance of a class

        Args:
            lag_days_list (list): a list of lag total sales of interest.
            column_name (str): Name of dataframe column to be referenced.

        Raises:
            None

        Returns:
            pd.DataFrame: Dataframe containing generated lag columns of specified feature of interest
        """
        logging.info("Start generating lag values")
        # Create a lagged dataframe for time period of interest
        lagged_df = self.lagged_proxy_revenue_df[[column_name]].copy()
        # Create empty dataframe to store data for inference
        result_df = pd.DataFrame(index=self.location_df.index)
        # get range of acceptable lags and check if input falls within the range.
        min_lag = (
            self.location_df.index[-1] - self.lagged_proxy_revenue_df.index[-1]
        ).days
        max_lag = (
            self.location_df.index[0] - self.lagged_proxy_revenue_df.index[0]
        ).days
        if self.check_range(lag_days_list, min_lag, max_lag) > 0:
            logging.error(
                f"Invalid lag value input. Accepted range is within {min_lag} to {max_lag}"
            )
        try:
            for lag_days in lag_days_list:
                # create new lag column name
                new_lag_column_name = "_".join(
                    ["lag", str(lag_days), "days", str(column_name)]
                )
                # get the start and end index to slice lagged_df
                start_index = self.location_df.index[0] - timedelta(days=lag_days)
                lagged_df[new_lag_column_name] = lagged_df[column_name].loc[
                    start_index : start_index
                    + timedelta(days=len(self.location_df) - 1)
                ]
                # add the non nan values (lagged proxy revenue) to the empty result_df as the newly created lagged columns are mapped to different dates, with the rest being nan values.
                result_df[new_lag_column_name] = lagged_df[
                    lagged_df[new_lag_column_name].notnull()
                ][new_lag_column_name].values

            logging.info("Lag values generated")
            logging.debug(
                f"Generated dataframe: {lagged_df.shape} with columns {lagged_df.columns}"
            )

            lagged_df.index = pd.to_datetime(lagged_df.index)
            lagged_df.drop(column_name, axis=1, inplace=True)
            return result_df

        except KeyError:
            log_string = f"Unable to access {column_name} column. Please check if column name is correct."
            logging.error(log_string, exc_info=True)
            return None

    def generate_sma_sales(
        self, window_days_list: list, column_name: str
    ) -> Union[None, pd.DataFrame]:
        """Function creates a dataframe containing simple moving average of a column of interest (column_name) according to the window length specified(window_days_list).

        Args:
            window_days_list (list): window length in days.
            column_name (str): Name of dataframe column to be referenced.

        Raises:
            None

        Returns:
            pd.DataFrame: dataframe with the simple moving average of sales
        """
        logging.info("Start generating simple moving average of sales")
        sma_df = self.lagged_proxy_revenue_df[[column_name]].copy()
        # get range of acceptable lags and check if input falls within the range.
        max_lag = len(self.lagged_proxy_revenue_df) - len(self.location_df) + 1
        if self.check_range(window_days_list, 2, max_lag) > 0:
            logging.error(
                f"Invalid lag value input. Accepted range is within 2 to {max_lag}"
            )
        try:
            for n_days in window_days_list:
                new_window_day_column_name = "_".join(
                    ["sma_window", str(n_days), "days", str(column_name)]
                )

                sma_df[new_window_day_column_name] = (
                    sma_df[column_name].rolling(n_days).mean()
                )
            # return the most recent days' SMA values for inference
            sma_df = sma_df.iloc[-len(self.location_df) :]
            sma_df.index = self.location_df.index
            logging.info("Simple moving average of sales generated")
            logging.debug(
                f"Generated dataframe: {sma_df.shape} with columns {sma_df.columns}"
            )

            # Drop original column used for simple moving average sales generation
            sma_df.drop(column_name, axis=1, inplace=True)
            sma_df.index = pd.to_datetime(sma_df.index)
            return sma_df

        except KeyError:
            log_string = f"Unable to access {column_name} column. Please check."
            logging.error(log_string, exc_info=True)
            return None

    def generate_lag_avg_weekly_sales(
        self, num_weeks_list: list, sales_column_name: str
    ) -> pd.DataFrame:
        """Function that generates the lag values of average weekly sales based on number of weeks defined in num_weeks_list.
        Args:
            num_weeks_list (list): List of weeks for lag values generations
            sales_column_name (str): Name of dataframe column related to sales.

        Raises:
            None

        Returns:
            pd.DataFrame: dataframe with the moving average of weekly sales
        """

        if not isinstance(num_weeks_list, list):
            log_string = f"num_weeks_list argument is not of list type, will be set to {MARKETING_SHEETS_LIST}"
            logging.info(log_string)
            num_weeks_list = MARKETING_SHEETS_LIST

        if not isinstance(sales_column_name, str):
            log_string = "sales_column_name argument is not of str type, will be typecasted to string."
            logging.info(log_string)
            sales_column_name = str(sales_column_name)

        # get range of acceptable lags and check if input falls within the range.
        max_week = len(self.lagged_proxy_revenue_df) // 7
        if self.check_range(num_weeks_list, 1, max_week) > 0:
            logging.error(
                f"Invalid lag value input. Accepted range is within 1 to {max_lag}"
            )

        logging.info("Generate moving average of weekly sales")

        # Generate group number based on day_of_week. Requires column that provides day of week information in format of monday, tuesday, ...

        logging.info(
            "Generating data groups based on specified start_day and days_per_group config values."
        )
        new_group_column_name = "group"

        # Extract only necessary features
        df = self.lagged_proxy_revenue_df[sales_column_name].copy().to_frame()
        df = get_group_number(
            df=df,
            new_column_name=new_group_column_name,
        )
        # Apply a groupby based on generated groups and get the mean value based on column of interest which is sales
        weekly_sales = df.groupby(new_group_column_name)[[sales_column_name]].mean()

        # create dataframe to store weekly sales
        result_df = pd.DataFrame(index=self.location_df.index)
        # Loop through num_weeks_list to generate new features
        index = 0
        for num_week in num_weeks_list:
            avg_weekly_proxy_revenue_df = weekly_sales.rolling(
                window=num_week, min_periods=1
            ).mean()
            logging.debug(
                f"Generating weekly rolling sales for {num_week} week window period."
            )

            # Generate rolling weekly sales column name for each num week
            new_rolling_weekly_sales_column_name = "_".join(
                ["lag", str(num_week), "week_mean_weekly", str(sales_column_name)]
            )
            result_df[
                new_rolling_weekly_sales_column_name
            ] = avg_weekly_proxy_revenue_df.iloc[index][0]
            index += 1
        logging.info(
            f"Moving average of weekly sales generated for all weeks specified in {num_weeks_list}"
        )

        return result_df

    def get_campaign_count(self) -> pd.DataFrame:
        """Function that generates the ongoing campaign counts for each day.

        Returns:
            pd.DataFrame: dataframe containing the campaign count for each day.
        """
        # Create a new dataframe
        mkt_counts_df = self.location_df[CAMPAIGN_COLUMN].copy().to_frame()
        # Get campaign count
        mkt_counts_df[CAMPAIGN_COUNT_COLUMN] = mkt_counts_df[CAMPAIGN_COLUMN].map(
            lambda x: x.count(",") + 1 if x != "No" else 0
        )
        logging.info("Total marketing content counts generated for all items in list")
        logging.debug(
            f"Generated dataframe: {mkt_counts_df.shape} with columns {mkt_counts_df.columns}"
        )
        mkt_counts_df.drop([CAMPAIGN_COLUMN], axis=1, inplace=True)
        mkt_counts_df.index = pd.to_datetime(mkt_counts_df.index)
        return mkt_counts_df

    def check_range(self, range_list: list, l: int, r: int) -> int:
        """Helper function to check if the list inputs are within a certain range.

        Args:
            range_list (list): list of inputs
            l (int): left lower bound
            r (int): right upper bound

        Returns:
            int: number of inputs which do not fall within the accepted range.
        """
        return len([x for x in range_list if x < l or x > r])
