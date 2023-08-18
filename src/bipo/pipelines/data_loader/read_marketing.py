from bipo.pipelines.data_loader.data_check import DataCheck
import pandas as pd
import logging

# pd.options.mode.chained_assignment = None  # suppress warning entirely
from datetime import timedelta
import numpy as np

import sys
from kedro.config import ConfigLoader
from kedro.framework.project import settings

# from ...utils import get_project_path  # , get_logger
from bipo.utils import get_project_path

# logging = logging.getLogger("kedro")
logging = logging.getLogger(__name__)

# initiate to constants.yml and parameters.yml
project_path = get_project_path()
conf_loader = ConfigLoader(conf_source=project_path / "conf")
conf_params = conf_loader["parameters"]["dataloader"]
constants = conf_loader.get("constants*")["dataloader"]

# MARKETING_FILE = params["marketing_file"]
EXPECTED_COLUMNS = constants["expected_columns"]
NEW_COLUMN_NAMES = constants["new_column_names"]
MARKETING_SHEET_CAMPAIGN = conf_params["marketing_sheet_campaign"]
MARKETING_SHEET_AD = conf_params["marketing_sheet_ad"]
DATE_RANGE = conf_params["date_range"]


class ReadMarketingData(DataCheck):
    """reads marketing campaign data into a dataframe.

    Inherits the data_check class.
    """

    def __init__(self):
        super().__init__()

    def read_campaign_data(self, file_path: str) -> pd.DataFrame:
        """Function to read marketing.xlsx "campaign", "promotion", "product launch" sheet, convert the dataframe with date range into a time series df and add all ongoing campaigns in each dates. Note, Columns name used are 'Name', 'Start Date' and 'End Date' column is used.

        Function includes check_file_format_encoding, data_exist and check_columns

        self.marketing_df_list_campaign will contain "campaign", "promotion", "product launch" dataframe.

        Args:
            file_path (str): file path to marketing.xlsx

        Returns:
            marketing_df_list_campaign (list): list that contain "campaign", "promotion", "product launch" dataframe.
        """
        self.check_file_format_encoding(file_path)

        self.marketing_df_list_campaign = []
        for sheet_name in MARKETING_SHEET_CAMPAIGN:
            # Read the data from specific sheet
            data_df = pd.read_excel(file_path, sheet_name=sheet_name)

            # Lowercase the sheet name and remove any spacing at the back
            lower_case_sheet_name = sheet_name.lower().rstrip()

            # Check if data have expected columns
            self.check_columns(data_df.columns.tolist(), lower_case_sheet_name)

            # Check if there is data inside
            self.check_data_exists(data_df)

            # Get the column of interests from each sheets
            column_list = constants["marketing_columns"][sheet_name]
            marketing_sheet_df = data_df[column_list]

            # Copy to suppress warning
            marketing_sheet_df = marketing_sheet_df.copy(deep=True)

            # Get the respective columns
            feature_of_interest_name = column_list[2]
            start_date_name = column_list[0]
            end_date_name = column_list[1]

            # Set NAN End Date with a new end date based on 3 mth from start date. eg Wingstreet Launch

            start_date = marketing_sheet_df.loc[
                marketing_sheet_df[end_date_name].isnull(), start_date_name
            ].copy(deep=True)

            end_date = pd.to_datetime(start_date) + timedelta(
                days=conf_params["days_to_end_campaign"]
            )

            # Copy to suppress warning
            end_date = end_date.copy(deep=True)

            marketing_sheet_df[end_date_name] = marketing_sheet_df[
                end_date_name
            ].fillna(end_date)

            # Convert 'Start Date' and 'End Date' columns to datetime
            marketing_sheet_df[start_date_name] = pd.to_datetime(
                marketing_sheet_df[start_date_name]
            )
            marketing_sheet_df[end_date_name] = pd.to_datetime(
                marketing_sheet_df[end_date_name]
            )

            # Create a new dataframe with datetime index ranging from 2021-01-01 to 2022-12-31
            index = pd.date_range(start=DATE_RANGE[0], end=DATE_RANGE[1], freq="D")
            new_marketing_sheet_df = pd.DataFrame(index=index)

            # Iterate over each row of the original dataframe
            new_marketing_sheet_df = self.set_campaign_names(
                sheet_name,
                data_df,
                new_marketing_sheet_df,
                start_date_name,
                end_date_name,
                feature_of_interest_name,
            )
            # for _, row in marketing_sheet_df.iterrows():
            #     start_date = row[start_date_name]
            #     end_date = row[end_date_name]
            #     feature_of_interest = str(row[feature_of_interest_name])

            #     # Check if end_data < start_date (move to datacheck?)
            #     if end_date < start_date:
            #         logging.error(
            #             f"{start_date} > {end_date} Please check end date of sheet: {sheet_name}, ID: {feature_of_interest}"
            #         )

            #     # Find the overlapping dates between the current row and the new dataframe
            #     overlap_dates = new_marketing_sheet_df.index[
            #         (new_marketing_sheet_df.index >= start_date)
            #         & (new_marketing_sheet_df.index <= end_date)
            #     ]

            #     # Store the names as the value in the 'ID' column for the overlapping dates
            #     for date in overlap_dates:
            #         if feature_of_interest_name not in new_marketing_sheet_df.columns:
            #             new_marketing_sheet_df[feature_of_interest_name] = np.nan

            #         if pd.isnull(
            #             new_marketing_sheet_df.loc[date, feature_of_interest_name]
            #         ):
            #             new_marketing_sheet_df.at[
            #                 date, feature_of_interest_name
            #             ] = feature_of_interest
            #         else:
            #             new_marketing_sheet_df.at[date, feature_of_interest_name] = (
            #                 new_marketing_sheet_df.at[date, feature_of_interest_name]
            #                 + ", "
            #                 + feature_of_interest
            #             )

            # Conver NaN to "No"
            new_marketing_sheet_df = new_marketing_sheet_df.fillna("No")

            # change the column names
            column_mapping = dict(
                zip(
                    EXPECTED_COLUMNS[lower_case_sheet_name],
                    NEW_COLUMN_NAMES[lower_case_sheet_name],
                )
            )
            new_marketing_sheet_df = new_marketing_sheet_df.rename(
                columns=column_mapping
            )
            self.marketing_df_list_campaign.append(new_marketing_sheet_df)
        # marketing_campaign_df = pd.concat(marketing_df_list_campaign, axis=1)
        return self.marketing_df_list_campaign

    def read_ad_data(self, file_path: str):
        """Function to read marketing.xlsx "TV Ad", "Radio Ad" "Poster Campaign", "Digital ", "Youtube Ad", "Instagram Ad", "Facebook Ad" sheets,
        Convert the dataframe with date range into a time series df and add all ongoing sheets in each dates. Note, Columns name used are 'ID', 'Start Date', 'End Date' and 'Cost' column.

        Function includes check_file_format_encoding, data_exist and check_columns

        self.marketing_df_list_ad will contain "TV Ad", "Radio Ad" "Poster Campaign", "Digital ", "Youtube Ad", "Instagram Ad", "Facebook Ad" dataframe.

        Args:
            file_path (str): file path to marketing.xlsx

        Returns:
            marketing_df_list_ad (list): list that contain "TV Ad", "Radio Ad" "Poster Campaign", "Digital ", "Youtube Ad", "Instagram Ad", "Facebook Ad" dataframe.
        """
        # Check file format encoding
        self.check_file_format_encoding(file_path)

        self.marketing_df_list_ad = []
        for sheet_name in MARKETING_SHEET_AD:
            data_df = pd.read_excel(file_path, sheet_name=sheet_name)

            # Check if there is data inside
            self.check_data_exists(data_df)

            # Lowercase the sheet name and remove the spacing at the back
            lower_case_sheet_name = sheet_name.lower().rstrip()

            # Check if data have expected columns
            self.check_columns(data_df.columns.tolist(), lower_case_sheet_name)

            data_df.replace("?", np.nan, inplace=True)

            # Replace irregular naming for 'Start Date ' and 'Start Date' columns
            data_df.columns = data_df.columns.str.rstrip()
            sheet_name = sheet_name.rstrip()

            # Get the column of interests from each sheets
            column_list = constants["marketing_columns"][sheet_name]
            marketing_sheet_df = data_df[column_list]

            # Get the respective columns
            feature_of_interest_name = column_list[2]
            cost_name = column_list[3]
            start_date_name = column_list[0]
            end_date_name = column_list[1]

            # Convert 'Start Date' and 'End Date' columns to datetime
            data_df[start_date_name] = pd.to_datetime(data_df[start_date_name])
            data_df[end_date_name] = pd.to_datetime(data_df[end_date_name])

            # Create a new dataframe with datetime index ranging from 2021-01-01 to 2022-12-31
            index = pd.date_range(start=DATE_RANGE[0], end=DATE_RANGE[1], freq="D")
            new_marketing_sheet_df = pd.DataFrame(index=index)

            # For "cost" column naming
            sheet_name_cost = lower_case_sheet_name.replace(" ", "_") + "_cost"

            # For imputation of "?" in "Cost" Columns
            data_df = self.get_and_impute_daily_cost(
                sheet_name,
                data_df,
                start_date_name,
                end_date_name,
                cost_name,
                feature_of_interest_name,
            )
            # daily_cost_list = []
            # for _, row in data_df.iterrows():
            #     start_date = row[start_date_name]
            #     end_date = row[end_date_name]
            #     total_cost = str(row[cost_name])
            #     feature_of_interest = str(row[feature_of_interest_name])

            #     # Some value in cost have 2 dots Eg: "$23.865.59". replace the  "."
            #     # Check if the string has at least two dots (move to datacheck?)
            #     if total_cost.count(".") >= 2:
            #         logging.error(
            #             f"{total_cost} has at least two dots. Please check  sheet: {sheet_name}, ID: {feature_of_interest}"
            #         )
            #         # Replace the first occurrence of "." with ","
            #         total_cost = float(total_cost.replace(".", "", 1).replace("$", ""))

            #     else:
            #         total_cost = float(total_cost)

            #     # Check if end_data < start_date (move to datacheck?)
            #     if end_date < start_date:
            #         daily_cost = np.nan  # add nan value as placeholder
            #         logging.error(
            #             f"{start_date} > {end_date} Please check end date of sheet: {sheet_name}, ID: {feature_of_interest}"
            #         )
            #     else:
            #         # calculate the daily cost from "Cost"
            #         date_range = pd.date_range(start=start_date, end=end_date)

            #         # Calculate the price divided by the length of the date range
            #         # note that for Digital ID 17, the end date is 25/8/2011
            #         daily_cost = round((total_cost / len(date_range)), 2)

            #     # append daily_cost into a empty list
            #     daily_cost_list.append(daily_cost)
            # # Add daily cost column to data_df
            # data_df["daily_cost"] = daily_cost_list

            # # remove NaN from daily_cost_list
            # daily_cost_list = list(filter(lambda x: not pd.isna(x), daily_cost_list))

            # # sort data before getting median
            # daily_cost_list = sorted(daily_cost_list)

            # # calculate the median daily cost
            # median_daily_cost = np.median(daily_cost_list)

            # # imputate median_daily_cost to replce "NaN" in "daily_cost" Column
            # data_df["daily_cost"] = data_df["daily_cost"].fillna(median_daily_cost)

            # Iterate over each row of the original dataframe
            new_marketing_sheet_df = self.set_daily_cost(
                data_df,
                new_marketing_sheet_df,
                start_date_name,
                end_date_name,
                feature_of_interest_name,
                sheet_name_cost,
            )
            # for _, row in data_df.iterrows():
            #     start_date = row[start_date_name]
            #     end_date = row[end_date_name]
            #     feature_of_interest = str(row[feature_of_interest_name])
            #     # cost = str(row[cost_name])
            #     cost_per_day = row["daily_cost"]

            #     # Find the overlapping dates between the current row and the new dataframe
            #     overlap_dates = new_marketing_sheet_df.index[
            #         (new_marketing_sheet_df.index >= start_date)
            #         & (new_marketing_sheet_df.index <= end_date)
            #     ]

            #     # Store the names as the value in the 'ID' column for the overlapping dates
            #     for date in overlap_dates:
            #         # Create a new column "ID" with all NaN values
            #         if feature_of_interest_name not in new_marketing_sheet_df.columns:
            #             new_marketing_sheet_df[feature_of_interest_name] = np.nan

            #         # If the cell is NaN, replace it with ID
            #         if pd.isna(
            #             new_marketing_sheet_df.at[date, feature_of_interest_name]
            #         ):
            #             new_marketing_sheet_df.at[
            #                 date, feature_of_interest_name
            #             ] = feature_of_interest

            #         # If the cell has a existing ID, append the any overlapping ID to the exisiting ID.
            #         else:
            #             new_marketing_sheet_df.at[date, feature_of_interest_name] = (
            #                 str(
            #                     new_marketing_sheet_df.at[
            #                         date, feature_of_interest_name
            #                     ]
            #                 )
            #                 + ", "
            #                 + feature_of_interest
            #             )

            #     for date in overlap_dates:
            #         # Create a new column "sheet_name_cost" with all 0
            #         if sheet_name_cost not in new_marketing_sheet_df.columns:
            #             new_marketing_sheet_df[sheet_name_cost] = 0

            #         # If the cell is 0, replace it with cost_per_day
            #         if new_marketing_sheet_df.loc[date, sheet_name_cost] == 0:
            #             new_marketing_sheet_df.at[date, sheet_name_cost] = cost_per_day

            #         # If the cell has a existing cost_per_day, sum the current cost_per_day with the new cost_per_day.
            #         else:
            #             total_cost = (
            #                 new_marketing_sheet_df.at[date, sheet_name_cost]
            #                 + cost_per_day
            #             )
            #             new_marketing_sheet_df.at[date, sheet_name_cost] = total_cost
            # rename the columns
            # change the column names
            column_mapping = dict(
                zip(
                    EXPECTED_COLUMNS[lower_case_sheet_name],
                    NEW_COLUMN_NAMES[lower_case_sheet_name],
                )
            )
            new_marketing_sheet_df = new_marketing_sheet_df.rename(
                columns=column_mapping
            )

            # Convert NaN to "No"
            new_marketing_sheet_df = new_marketing_sheet_df.fillna("No")

            self.marketing_df_list_ad.append(new_marketing_sheet_df)

        return self.marketing_df_list_ad

    def merge_marketing(self, marketing_df_list_campaign, marketing_df_list_ad):
        """Merge both marketing_df_list_campaign and marketing_df_list_ad into marketing_df

        Args:
            marketing_df_list_campaign (list): list that contain "Campaign", "Promotions", "Product Launch" dataframe.
            marketing_df_list_ad (list): list that contain "TV Ad", "Radio Ad" "Poster Campaign", "Digital ", "Youtube Ad", "Instagram Ad", "Facebook Ad" dataframe.

        Returns:
            marketing_df(pd.DataFrame): merged marketing dataframe with all the sheets except "Influencer Engagement"
        """
        merged_list = marketing_df_list_ad + marketing_df_list_campaign
        marketing_df = pd.concat(merged_list, axis=1)

        return marketing_df

    def get_and_impute_daily_cost(
        self,
        sheet_name,
        data_df,
        start_date_name,
        end_date_name,
        cost_name,
        feature_of_interest_name,
    ):
        """Function to get daily cost and add a "daily_cost" column to data_df. Impute any missing "daily_cost" with the median. Return data_df with "daily_cost" column and imputed values.


        Args:
            data_df (pd.DataFrame): original marketing dataframe

        Returns:
            Return: Imputed data_df
        """
        # Get the daily cost
        # For imputation of "?" in "Cost" Columns
        daily_cost_list = []
        for _, row in data_df.iterrows():
            start_date = str(row[start_date_name])
            end_date = str(row[end_date_name])
            total_cost = str(row[cost_name])
            feature_of_interest = str(row[feature_of_interest_name])

            # Some value in cost have 2 dots Eg: "$23.865.59". replace the  "."
            # Check if the string has at least two dots (move to datacheck?)
            if total_cost.count(".") >= 2:
                logging.error(
                    f"{total_cost} has at least two dots. Please check  sheet: {sheet_name}, ID: {feature_of_interest}"
                )
                # Replace the first occurrence of "." with ","
                total_cost = float(total_cost.replace(".", "", 1).replace("$", ""))

            else:
                total_cost = float(total_cost)

            # Check if end_data < start_date (move to datacheck?)
            if end_date < start_date:
                daily_cost = np.nan  # add nan value as placeholder
                logging.error(
                    f"{start_date} > {end_date} Please check end date of sheet: {sheet_name}, ID: {feature_of_interest}"
                )
            else:
                # calculate the daily cost from "Cost"
                date_range = pd.date_range(start=start_date, end=end_date)

                # Calculate the price divided by the length of the date range
                # note that for Digital ID 17, the end date is 25/8/2011
                daily_cost = round((total_cost / len(date_range)), 2)

            # append daily_cost into a empty list
            daily_cost_list.append(daily_cost)
        # Add daily cost column to data_df
        data_df["daily_cost"] = daily_cost_list

        # remove NaN from daily_cost_list
        daily_cost_list = list(filter(lambda x: not pd.isna(x), daily_cost_list))

        # sort data before getting median
        daily_cost_list = sorted(daily_cost_list)

        # calculate the median daily cost
        median_daily_cost = np.median(daily_cost_list)

        # imputate median_daily_cost to replce "NaN" in "daily_cost" Column
        data_df["daily_cost"] = data_df["daily_cost"].fillna(median_daily_cost)

        return data_df

    def set_daily_cost(
        self,
        data_df,
        new_marketing_sheet_df,
        start_date_name,
        end_date_name,
        feature_of_interest_name,
        sheet_name_cost,
    ):
        """Function to convert data range into datetime index, set "daily_cost" from data_df into datatime index format. For overlapping campaigns, the daily cost from both campaigns will summed, the feature_of_interest will be appended in a string and save in new_marketing_sheet_df.

        Args:
            data_df (pd.DataFrame): original df with "daily_cost" column
            new_marketing_sheet_df (pd.DataFrame): empty dataframe with preset datatime index

        Returns:
            new_marketing_sheet_df
        """
        # Iterate over each row of the original dataframe
        for _, row in data_df.iterrows():
            start_date = row[start_date_name]
            end_date = row[end_date_name]
            feature_of_interest = str(row[feature_of_interest_name])
            # cost = str(row[cost_name])
            cost_per_day = row["daily_cost"]

            # Find the overlapping dates between the current row and the new dataframe
            overlap_dates = new_marketing_sheet_df.index[
                (new_marketing_sheet_df.index >= start_date)
                & (new_marketing_sheet_df.index <= end_date)
            ]

            # Store the names as the value in the 'ID' column for the overlapping dates
            for date in overlap_dates:
                # Create a new column "ID" with all NaN values
                if feature_of_interest_name not in new_marketing_sheet_df.columns:
                    new_marketing_sheet_df[feature_of_interest_name] = np.nan

                # If the cell is NaN, replace it with ID
                if pd.isna(new_marketing_sheet_df.at[date, feature_of_interest_name]):
                    new_marketing_sheet_df.at[
                        date, feature_of_interest_name
                    ] = feature_of_interest

                # If the cell has a existing ID, append the any overlapping ID to the exisiting ID.
                else:
                    new_marketing_sheet_df.at[date, feature_of_interest_name] = (
                        str(new_marketing_sheet_df.at[date, feature_of_interest_name])
                        + ", "
                        + feature_of_interest
                    )

            for date in overlap_dates:
                # Create a new column "sheet_name_cost" with all 0
                if sheet_name_cost not in new_marketing_sheet_df.columns:
                    new_marketing_sheet_df[sheet_name_cost] = 0

                # If the cell is 0, replace it with cost_per_day
                if new_marketing_sheet_df.loc[date, sheet_name_cost] == 0:
                    new_marketing_sheet_df.at[date, sheet_name_cost] = cost_per_day

                # If the cell has a existing cost_per_day, sum the current cost_per_day with the new cost_per_day.
                else:
                    total_cost = (
                        new_marketing_sheet_df.at[date, sheet_name_cost] + cost_per_day
                    )
                    new_marketing_sheet_df.at[date, sheet_name_cost] = total_cost

        return new_marketing_sheet_df

    def set_campaign_names(
        self,
        sheet_name,
        data_df,
        new_marketing_sheet_df,
        start_date_name,
        end_date_name,
        feature_of_interest_name,
    ):
        """Function to convert data range into datetime index, set "Name" from data_df into datatime index format. For overlapping campaigns, the Campaign names will be stacked in a string and save in new_marketing_sheet_df.

        Args:
            data_df (pd.DataFrame): original df
            new_marketing_sheet_df (pd.DataFrame): empty dataframe with preset datatime index

        Returns:
            new_marketing_sheet_df
        """
        # Iterate over each row of the original dataframe
        for _, row in data_df.iterrows():
            start_date = row[start_date_name]
            end_date = row[end_date_name]
            feature_of_interest = str(row[feature_of_interest_name])

            # Check if end_data < start_date (move to datacheck?)
            if end_date < start_date:
                logging.error(
                    f"{start_date} > {end_date} Please check end date of sheet: {sheet_name}, ID: {feature_of_interest}"
                )

            # Find the overlapping dates between the current row and the new dataframe
            overlap_dates = new_marketing_sheet_df.index[
                (new_marketing_sheet_df.index >= start_date)
                & (new_marketing_sheet_df.index <= end_date)
            ]

            # Store the names as the value in the 'ID' column for the overlapping dates
            for date in overlap_dates:
                if feature_of_interest_name not in new_marketing_sheet_df.columns:
                    new_marketing_sheet_df[feature_of_interest_name] = np.nan

                if pd.isnull(
                    new_marketing_sheet_df.loc[date, feature_of_interest_name]
                ):
                    new_marketing_sheet_df.at[
                        date, feature_of_interest_name
                    ] = feature_of_interest
                else:
                    new_marketing_sheet_df.at[date, feature_of_interest_name] = (
                        new_marketing_sheet_df.at[date, feature_of_interest_name]
                        + ", "
                        + feature_of_interest
                    )

        return new_marketing_sheet_df
