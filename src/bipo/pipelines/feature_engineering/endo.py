import pandas as pd
import os
import numpy as np
import warnings
from datetime import datetime, date
import sys
from pathlib import Path
from kedro.config import ConfigLoader

# from .common

from bipo.pipelines.feature_engineering.common import (
    read_csv_file,
    get_group_number,
    apply_mean_std_binning,
    apply_equal_freq_binning,
)


sys.dont_write_bytecode = True

from typing import Union
from bipo.utils import get_project_path
import logging

logging = logging.getLogger(__name__)
# Instantiate config
project_path = get_project_path()
conf_loader = ConfigLoader(conf_source=project_path / "conf")
constants = conf_loader.get("constants*")
conf_params = conf_loader["parameters"]

# load configs
# FROM DATALOADER MODULE

DATALOADER_SOURCE_DATA_DIR = constants["dataloader"]["data_source_dir"]
MARKETING_FILE = constants["dataloader"]["marketing_file"]
MARKETING_SHEETS_LIST = constants["dataloader"]["marketing_sheets"]
CAMPAIGN_SHEET = constants["dataloader"]["campaign_sheet"]
DAY_OF_WEEK_COLUMN_NAME = constants["dataloader"]["new_column_names"]["holiday"][-1]

# FROM FEATURE ENGINEERING MODULE
LAG_LIST = conf_params["feature_engineering"]["endo"]["lag_days_list"]
WINDOW_LIST = conf_params["feature_engineering"]["endo"]["window_days_list"]
NUM_LIST = conf_params["feature_engineering"]["endo"]["num_weeks_list"]
SHIFT_PERIOD = conf_params["feature_engineering"]["endo"]["shift_period"]
BIN_APPROACH = conf_params["feature_engineering"]["endo"]["bin_approach"]
BIN_LABELS_LIST = conf_params["feature_engineering"]["endo"]["bin_labels_list"]
TARGET_FEATURE = conf_params["general"]["target_feature"]


class Endogenous:
    """Feature generated for endogenous features:
    1. Lagged values: lag of 1,2 and 7

    """

    def __init__(self, outlet_df: pd.DataFrame):
        """Initializes the Exog class.

        Args:
            outlet_df (pd.DataFrame): Individual outlet dataframe.
        """
        # Variables to store artefacts on feature engineer works
        self.added_features_endo = []
        self.endo_df = outlet_df

        return None

    def get_added_features_endo(self) -> list:
        """Getter function for added_feature_endo.

        Args:
            None

        Raises:
            None

        Returns:
            list: Instance variable self.added_features_endo
        """
        return self.added_features_endo

    # merge function here
    def endo_transform(self) -> Union[None, pd.DataFrame]:
        """Run the entire endogenous feature engineering pipeline.

        Args:
            None

        Raises:
            None

        Returns:
            pd.DataFrame: Dataframe with only newly engineered endogenous features.
        """

        # Construct dataframes representing only transformed variables
        lagged_df = self.generate_lag(
            lag_days_list=LAG_LIST, column_name=TARGET_FEATURE
        )
        sma_df = self.generate_sma_sales(
            window_days_list=WINDOW_LIST,
            column_name=TARGET_FEATURE,
            shift_period=SHIFT_PERIOD,
        )
        weekly_average_df = self.generate_lag_avg_weekly_sales(
            num_weeks_list=NUM_LIST,
            sales_column_name=TARGET_FEATURE,
            shift_period=SHIFT_PERIOD,
        )

        # Create boolean columns for dates indicating if there is start of end of event using original marketing file
        raw_marketing_filepath = os.path.join(
            *DATALOADER_SOURCE_DATA_DIR, MARKETING_FILE
        )
        mkt_start_end_df = self.generate_marketing_start_end_indication(
            filepath=raw_marketing_filepath, excel_sheet_list=[CAMPAIGN_SHEET]
        )

        mkt_start_end_df.index = pd.to_datetime(mkt_start_end_df.index)

        # Create marketing count and existence related information. Will be None if self.mkt_type_categories_list = None as they depend on it for processing
        active_mkt_counts_df = self.get_marketing_content_counts(
            marketing_sheet_list=[CAMPAIGN_SHEET]
        )
        active_mkt_status_df = self.get_marketing_activity_status(
            marketing_sheet_name=CAMPAIGN_SHEET
        )

        campaign_daily_cost_df = self.get_campaign_daily_cost()

        # Generate bin for target feature based on bin approach stated in data_conf.yml
        try:
            bin_df = self.generate_bins(
                column_name=TARGET_FEATURE,
                bin_approach=BIN_APPROACH,
                bin_labels_list=BIN_LABELS_LIST,
            )
        except:
            bin_df = self.endo_df[TARGET_FEATURE].to_frame()
            logging.info("Unable to bin target feature")
            return None

        # generate feature is_weekend
        is_weekend_df = self.convert_to_weekend()

        try:
            df_merge_list = [
                lagged_df,
                sma_df,
                weekly_average_df,
                mkt_start_end_df,
                active_mkt_counts_df,
                active_mkt_status_df,
                is_weekend_df,
                campaign_daily_cost_df,
            ]

            # Base copy for use in merging
            endo_concat_df = bin_df.copy()
            endo_concat_df.index = pd.to_datetime(endo_concat_df.index)

            # Merge dataframe, when merging via index, it is important to ensure the index are of same time to avoid merged column values from changing to Nan.
            for df in df_merge_list:
                df.index = pd.to_datetime(df.index)
                endo_concat_df = endo_concat_df.merge(
                    df, how="left", left_index=True, right_index=True
                )

        except ValueError:
            log_string = (
                "Encountered all None when merging. Proceed with None as return"
            )
            logging.error(log_string)
            endo_concat_df = None

        logging.debug(f"Source dataframe contains {self.endo_df.shape[1]} features")

        logging.debug(f"Added total {len(self.added_features_endo)} features")

        logging.debug(
            f"Concatenated endogenous dataframe shape: {endo_concat_df.shape}\n"
        )

        return endo_concat_df

    def generate_lag(
        self, lag_days_list: list, column_name: str
    ) -> Union[None, pd.DataFrame]:
        """Creates a dataframe containing generated lag values based on column_name variable representing dataframe column name according to input lag_days_list from a dataframe instance of a class

        Args:
            lag_list_days (list): a list of lag total sales of interest
            column_name (str): Name of dataframe column to be referenced.

        Raises:
            None

        Returns:
            pd.DataFrame: Dataframe containing generated lag columns of specified feature of interest
        """
        logging.info("Start generating lag values")

        # Create a lagged dataframe for time period of interest
        lagged_df = self.endo_df[[column_name]].copy()

        try:
            for lag_days in lag_days_list:
                new_lag_column_name = "_".join(
                    ["lag", str(lag_days), "days", str(column_name)]
                )
                lagged_df[new_lag_column_name] = lagged_df[column_name].shift(
                    periods=lag_days
                )

                # Update attribute on created feature
                self.added_features_endo.extend([new_lag_column_name])

            logging.info("Lag values generated")
            logging.debug(
                f"Generated dataframe: {lagged_df.shape} with columns {lagged_df.columns}"
            )

            lagged_df.index = pd.to_datetime(lagged_df.index)
            lagged_df.drop(column_name, axis=1, inplace=True)
            return lagged_df

        except KeyError:
            log_string = f"Unable to access {column_name} column. Please check if column name is correct."
            logging.error(log_string, exc_info=True)
            return None

        except ValueError:
            log_string = (
                "Invalid values encountered, unable to make proper calculation  "
            )
            logging.error(log_string, exc_info=True)
            return None

    def generate_sma_sales(
        self, window_days_list: list, column_name: str, shift_period: int
    ) -> Union[None, pd.DataFrame]:
        """Function creates a dataframe containing simple moving average of a column of interest (column_name) according to the window length specified(window_days_list).

        Args:
            window_days_list (list): window length in days
            column_name (str): Name of dataframe column to be referenced.
            shift_period (int): Number of days to shift dataframe which is equal to the earliest lagged number of days for inference (difference between last inference date and last provided lagged date)

        Raises:
            None

        Returns:
            pd.DataFrame: dataframe with the simple moving average of sales
        """
        logging.info("Start generating simple moving average of sales")
        sma_df = self.endo_df[[column_name]].copy()

        try:
            for n_days in window_days_list:
                new_window_day_column_name = "_".join(
                    ["sma_window", str(n_days), "days", str(column_name)]
                )

                sma_df[new_window_day_column_name] = (
                    sma_df[column_name].rolling(n_days).mean()
                )

                # Update attribute on created feature
                self.added_features_endo.extend([new_window_day_column_name])
            sma_df.shift(periods=shift_period)

            logging.info("Simple moving average of sales generated")

            logging.debug(
                f"Generated dataframe: {sma_df.shape} with columns {sma_df.columns}"
            )

            # Drop original column used for simple moving average sales generation
            sma_df.drop(column_name, axis=1, inplace=True)
            sma_df.index = pd.to_datetime(sma_df.index)
            return sma_df

        except KeyError:
            log_string = f"Unable to access either of {reference_column_list} columns.Please check."
            logging.error(log_string, exc_info=True)
            return None

        except ValueError:
            log_string = (
                "Invalid values encountered, unable to make proper calculation  "
            )
            logging.error(log_string, exc_info=True)
            return None

    def generate_lag_avg_weekly_sales(
        self, num_weeks_list: list, sales_column_name: str, shift_period: int
    ) -> pd.DataFrame:
        """Function that generates the lag values of average weekly sales based on groupings defined in num_weeks_list.

        Args:
            num_weeks_list (list): List of weeks for lag values generations
            sales_column_name (str): Name of dataframe column related to sales.
            shift_period (int): Number of days to shift dataframe which is equal to the earliest lagged number of days for inference (difference between last inference date and last provided lagged date)

        Raises:
            None

        Returns:
            pd.DataFrame: dataframe with the moving average of weekly sales
        """

        if not isinstance(num_weeks_list, list):
            log_string = (
                f"num_weeks_list argument is not of list type, will be set to [1,2]"
            )
            logging.info(log_string)
            num_weeks_list = [1, 2]

        if not isinstance(sales_column_name, str):
            log_string = "sales_column_name argument is not of str type, will be typecasted to string."
            logging.info(log_string)
            sales_column_name = str(sales_column_name)

        logging.info("Generate moving average of weekly sales")

        # Generate group number based on day_of_week. Requires column that provides day of week information in format of monday, tuesday, ...

        logging.info(
            "Generating data groups based on specified start_day and days_per_group config values."
        )
        new_group_column_name = "group"

        # Extract only necessary features
        df = self.endo_df[[sales_column_name]].copy()
        # Update dataframe with interim column containing group information.
        # This is to facilitate lagged average sales based on X days grouped feature generation and would not be tracked as part of feature engineering

        df = get_group_number(
            df=df,
            new_column_name=new_group_column_name,
        )

        # Apply a groupby based on generated groups and get the mean value based on column of interest which is sales
        weekly_sales = df.groupby(new_group_column_name)[[sales_column_name]].mean()

        # Create a dataframe spanning over a time period based on source dataframe
        weekly_average_df = pd.DataFrame(
            pd.date_range(start=self.endo_df.index.min(), end=self.endo_df.index.max()),
            columns=["date"],
        )
        # Loop through num_weeks_list to generate new features
        for num_week in num_weeks_list:
            logging.debug(
                f"Generating weekly rolling sales for {num_week} week window period."
            )

            # Create a mapping dict that stores the rolling mean over past window period and shift the values by 1 time period
            mapping_dict = (
                weekly_sales.rolling(window=num_week).mean().shift(1).to_dict()
            )

            # Generate rolling weekly sales column name for each num week
            new_rolling_weekly_sales_column_name = "_".join(
                ["lag", str(num_week), "week_mean_weekly", str(sales_column_name)]
            )

            each_weekly_average_df = pd.DataFrame(
                {
                    new_rolling_weekly_sales_column_name: df[new_group_column_name].map(
                        mapping_dict[sales_column_name]
                    )
                }
            )

            each_weekly_average_df.index = pd.to_datetime(each_weekly_average_df.index)
            each_weekly_average_df.index.name = "date"
            # weekly_average_df = weekly_average_df.merge(
            #     each_weekly_average_df, left_index=True, right_index=True
            # )
            weekly_average_df = weekly_average_df.merge(
                each_weekly_average_df, how="left", on="date"
            )

            # Update attribute on created feature
            self.added_features_endo.extend([new_rolling_weekly_sales_column_name])

        logging.info(
            f"Moving average of weekly sales generated for all weeks specified in {num_weeks_list}"
        )
        # Set date as index
        weekly_average_df.set_index("date", inplace=True)
        weekly_average_df.index = pd.to_datetime(weekly_average_df.index)
        # shift the dataframe to match the inference period, where the period to shift is the difference between shift_period and the number of days per group ie. 7 for weekly.
        period = shift_period - 7
        weekly_average_df = weekly_average_df.shift(period)
        logging.debug(
            f"Generated dataframe: {weekly_average_df.shape} with columns {weekly_average_df.columns}"
        )
        return weekly_average_df

    def generate_marketing_start_end_indication(
        self, filepath: str, excel_sheet_list: list
    ) -> pd.DataFrame:
        """Reads the raw marketing data and generate boolean status of whether there is a start/end of a marketing event

        This is based on campaign information as well as its breakdown of sub-campaigns via promotion or product launch.

        Args:
            filepath (str): File path to xlsx containing marketing information. Expect xlsx file.
            excel_sheet_list (list): A list of marketing sheet of interest from xlsx file. (eg, [Campaign, Promotions, Product Launch])

        Returns:
            pd.DataFrame: dataframe with True or False for start and end date of each campaigns, promotion or product launch.
        """
        if not isinstance(filepath, str):
            log_string = (
                "filepath argument is not of str type, will be typecasted to str."
            )
            logging.info(log_string)
            filepath = str(filepath)

        if not isinstance(excel_sheet_list, list):
            log_string = f"excel_sheet_list argument is not of list type, will be set to ['Campaign']"
            logging.info(log_string)
            excel_sheet_list = ["Campaign"]

        logging.info("Start generating start and end dates of campaigns")
        marketing_start_end_list = []  # list to merge all the df

        mkt_events_df = pd.DataFrame(
            pd.date_range(start=self.endo_df.index.min(), end=self.endo_df.index.max()),
            columns=["date"],
        )
        for sheet in excel_sheet_list:
            # read the specific raw marketing sheet

            marketing_sheet_df = pd.read_excel(filepath, sheet_name=sheet)

            # Filter and get subset dataframe based on column names with "date" or "Date" as column header for start and end date in dataframe. Assuming it is start and end ordering

            start_date = marketing_sheet_df.filter(like="Date").columns[0]
            end_date = marketing_sheet_df.filter(like="Date").columns[1]

            # Construct new column name for date start and date end columns identified as cat_mkt_<sheet_name>_start and cat_mkt_<sheet_name>_end respectively. Subsequently they will be used to store datettime information of the start date and end date
            marketing_start = (
                "_".join(["cat_mkt", sheet, "start"]).lower().replace(" ", "_")
            )

            marketing_end = (
                "_".join(["cat_mkt", sheet, "end"]).lower().replace(" ", "_")
            )

            # convert date information to datetime under new column names
            marketing_sheet_df[marketing_start] = pd.to_datetime(
                marketing_sheet_df[start_date]
            )
            marketing_sheet_df[marketing_end] = pd.to_datetime(
                marketing_sheet_df[end_date]
            )

            specific_dates_start = set(marketing_sheet_df[marketing_start])
            specific_dates_end = set(marketing_sheet_df[marketing_end])

            # Set markers on start and end dates only with new column name
            mkt_events_df[marketing_start] = mkt_events_df["date"].map(
                lambda x: True if x in specific_dates_start else False
            )

            self.added_features_endo.extend([marketing_start])

            mkt_events_df[marketing_end] = mkt_events_df["date"].map(
                lambda x: True if x in specific_dates_end else False
            )

            self.added_features_endo.extend([marketing_end])

        # marketing_start_end_df = pd.concat(marketing_start_end_list, axis=1)
        # return marketing_start_end_df
        mkt_events_df.set_index("date", inplace=True)
        mkt_events_df.index = pd.to_datetime(mkt_events_df.index)

        logging.info("Boolean states for markering start and end dates generated")

        logging.debug(
            f"Generated dataframe: {mkt_events_df.shape} with columns {mkt_events_df.columns}"
        )

        # Return concatenated dataframe
        return mkt_events_df

    def get_marketing_content_counts(
        self, marketing_sheet_list: list
    ) -> Union[None, pd.DataFrame]:
        """Function that calculates the number of marketing contents information occurring per day(e.g no. of campaigns, no. of promotions, no. of products) based on relevant identified with cat_mkt_<name> columns in the class dataframe. A corresponding column will be created to indicate such counts.

        Args:
            marketing_sheet_list (str): A list of marketing sheet names from data_loader's marketing_sheets config file. Example: ["Campaign","Promotions" or "Product Launch"]

        Raises:
            None

        Returns:
            pd.DataFrame: dataframe containing count for campaigns, promotions or product launch for each date.
            None: if all the relevant columns for these are marketing channels are unavailable.
        """

        logging.info(
            "Generating marketing counts for each item for marketing related content"
        )

        # Columns to read
        reference_column_list = [
            "_".join(["cat_mkt", mkt_sheet.lower().rstrip().replace(" ", "_")])
            for mkt_sheet in marketing_sheet_list
        ]

        # New column
        new_count_column_list = [
            "_".join(["count_mkt", mkt_sheet.lower().rstrip().replace(" ", "_")])
            for mkt_sheet in marketing_sheet_list
        ]

        # Create a new dataframe
        mkt_counts_df = self.endo_df[reference_column_list].copy()
        # Get number of elements count

        try:
            mkt_counts_df[new_count_column_list] = self.endo_df[
                reference_column_list
            ].applymap(
                lambda x: x.count(",") + 1 if x != "No" else 0, na_action="ignore"
            )

            logging.info(
                "Total marketing content counts generated for all items in list"
            )
            logging.debug(
                f"Generated dataframe: {mkt_counts_df.shape} with columns {mkt_counts_df.columns}"
            )

            self.added_features_endo.extend(new_count_column_list)
            mkt_counts_df.drop(reference_column_list, axis=1, inplace=True)
            mkt_counts_df.index = pd.to_datetime(mkt_counts_df.index)
            return mkt_counts_df

        except KeyError:
            log_string = (
                f"Unable to access either columns of {reference_column_list} columns."
            )
            logging.error(log_string, exc_info=True)
            return None

        except ValueError:
            log_string = (
                "Invalid values encountered, unable to make proper calculation  "
            )
            logging.error(log_string, exc_info=True)
            return None

    def get_marketing_activity_status(
        self, marketing_sheet_name: str = CAMPAIGN_SHEET
    ) -> Union[None, pd.DataFrame]:
        """Function that generates True or False values based on existence of a marketing campaign found in class dataframe instance.


        Args:
            marketing_sheet_name (str, Optional): Sheet name of the excel file that is referenced via through config file. Defaults to Campaign

        Raises:
            None

        Returns:
            pd.DataFrame: dataframe containing boolean state of existence for campaigns for each date.
            None: if all the relevant columns for these are marketing channels are unavailable or wrongly referenced.
        """
        logging.info(
            "Generating marketing counts for each item for marketing related content"
        )

        # Column name to reference. Should be already in self.endo_df
        reference_column = "_".join(
            ["cat_mkt", marketing_sheet_name.lower().replace(" ", "_")]
        )

        # Column name to create for capturing marketing status
        new_status_column = "_".join(
            ["is_having", marketing_sheet_name.lower().replace(" ", "_")]
        )

        # Create a new dataframe containing daily dates
        mkt_status_df = self.endo_df[[reference_column]].copy()

        # Get activity status and apply mapping
        try:
            mkt_status_df[new_status_column] = mkt_status_df[reference_column].map(
                lambda x: False if x == "No" else True,
            )

            logging.info("Status of daily campaigns generated.")

            logging.debug(
                f"Generated dataframe: {mkt_status_df.shape} with columns {mkt_status_df.columns}"
            )
            self.added_features_endo.extend([new_status_column])
            mkt_status_df.index = pd.to_datetime(mkt_status_df.index)
            return mkt_status_df

        except KeyError:
            log_string = f"Column {reference_column} is not in class's dataframe, processed would be skipped."
            logging.error(log_string, exc_info=True)
            return None

        except ValueError:
            log_string = f"Invalid values encountered under column {reference_column}, unable to make proper calculation, processed would be skipped."
            logging.error(log_string, exc_info=True)
            return None

    def generate_bins(
        self, column_name: str, bin_approach: str, bin_labels_list: list
    ) -> pd.DataFrame:
        """Function that generates binned target feature according to the bin_approach.

        Args:
            column_name (str): Column name of dataframe for generate bins.
            bin_approach (str): Either equal frequency binning or mean std binning, which returns 4 bins.
            bin_labels_list (list): List containing bin labels as part of bin generation.

        Raises:
            None

        Returns:
            pd.DataFrame: Dataframe containing binned feature.
        """
        # Construct new column name for feature to be binned
        bin_column_name = f"binned_{column_name}"

        if bin_approach == "equal frequency":
            bin_df = apply_equal_freq_binning(
                df=self.endo_df[[column_name]].copy(),
                column_name=column_name,
                bin_column_name=bin_column_name,
                bin_labels_list=bin_labels_list,
            )

        elif bin_approach == "mean std":
            bin_df = apply_mean_std_binning(
                df=self.endo_df[[column_name]].copy(),
                column_name=column_name,
                bin_column_name=bin_column_name,
            )

        # Use equal frequency as default if bin approach is of some other specification
        else:
            bin_df = apply_equal_freq_binning(
                df=self.endo_df[[column_name]].copy(),
                column_name=column_name,
                bin_column_name=bin_column_name,
                bin_labels_list=bin_labels_list,
            )

        logging.debug(
            f"Generated dataframe: {bin_df.shape} with columns {bin_df.columns}"
        )
        self.added_features_endo.extend([bin_column_name])
        return bin_df

        # convert 'cat_day_of_week' to 'is_weekend' (sat, sun is weekend)

    def convert_to_weekend(self):
        """Function that generates is_weekend feature which has True for weekend and False for weekdays. Mon-Fri are weekdays, Sat-Sun are weekends.

        Args:
            None

        Raises:
            None

        Returns:
            pd.DataFrame: Dataframe containing is_weekend feature.
        """

        def check_weekend(day: str) -> bool:
            """Function that generates binned target feature according to the bin_approach.

            Args:
                day (str): day of week

            Raises:
                None

            Returns:
                bool: True if it is weekend, else False.
            """
            if day in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]:
                return False
            elif day in ["Saturday", "Sunday"]:
                return True

        is_weekend_df = self.endo_df["cat_day_of_week"].apply(check_weekend).to_frame()
        is_weekend_df.rename(columns={"cat_day_of_week": "is_weekend"}, inplace=True)
        self.added_features_endo.extend(["is_weekend"])
        return is_weekend_df

    def get_campaign_daily_cost(self) -> pd.DataFrame:
        """Function that generates the campaign daily cost by adding all the individual daily channel costs (e.g radio, tv ad cost)

        Returns:
            pd.DataFrame: Dataframe containing the campaign daily cost.
        """
        logging.info("Generating campaign daily cost")

        mkt_cost_df = self.endo_df.filter(like="_cost").copy()
        mkt_cost_df["campaign_daily_cost"] = mkt_cost_df.sum(axis=1)
        mkt_cost_df = mkt_cost_df["campaign_daily_cost"]
        return mkt_cost_df
