# """
# This is a boilerplate pipeline 'data_processing'
# generated using Kedro 0.18.10
# """

# # Import standard modules
# import pandas as pd
# import numpy as np
# from typing import Dict
# import logging

# # Import local modules
# # This will get the active logger during run time
# import sys

# sys.path.append("../../")
# sys.dont_write_bytecode = True

# from ...utils import get_project_path  # , get_logger

# # Third Party Imports
# from kedro.config import ConfigLoader

# # initiate to constants.yml and parameters.yml
# project_path = get_project_path()
# conf_loader = ConfigLoader(conf_source=project_path / "conf")
# conf_params = conf_loader["parameters"]["data_preprocessing"]
# constants = conf_loader.get("constants*")["data_preprocessing"]

# # load configs
# expected_dtypes = constants["expected_dtypes"]
# non_negative_exogeneous_columns = constants["non_negative_exogeneous_columns"]
# targe_variable = conf_loader.get("constants*")["general"]["target_feature"]

# logging = logging.getLogger("kedro")


# class DataPreprocessing:
#     """A class for preprocessing data through various checks and handling.

#     This includes:
#     - Checking and handling missing values
#     - Checking and casting data types
#     - Rectifying logic errors in sales/transactions
#     - Ensuring data meets certain constraints

#     """

#     def __init__(self) -> None:
#         self.merged_df = pd.DataFrame()

#     def run_pipeline(self, merged_df: pd.DataFrame) -> pd.DataFrame:
#         """Runs a series of preprocessing steps on the input DataFrame.

#         Args:
#             merged_df (pd.DataFrame): Input DataFrame to be preprocessed.

#         Returns:
#             pd.DataFrame: Processed DataFrame.
#         """

#         self.merged_df = merged_df.copy()
#         # Drop unnecessary columns
#         logging.info("Starting drop unnecessary columns")
#         self.merged_df.drop(
#             columns=conf_params["drop_columns"],
#             axis=1,
#             inplace=True,
#         )
#         logging.info("Removed unnecessary columns done")

#         # Replace special characters with NaNs
#         self.merged_df.replace({"-": np.nan, "?": np.nan, "[]": np.nan}, inplace=True)

#         # Checking and handling data types
#         logging.info("Starting data type checks")
#         self.check_data_types()
#         logging.info("Completed data type checks")

#         # Handling missing values
#         logging.info("Starting missing value checks")
#         self.check_and_handle_missing_values()
#         logging.info("Completed missing value checks")

#         # Checking and handling data constraints
#         logging.info("Starting data constraints checks")
#         self.check_and_handle_data_constraints()
#         logging.info("Completed data constraints checks")

#         processed_df = self.merged_df

#         return processed_df

#     def check_and_handle_missing_values(self) -> None:
#         """Checks for missing values in the DataFrame.

#         Columns with more than 50% missing values are dropped.
#         Remaining missing values are imputed using either median (numeric columns)
#         or mode (non-numeric columns).

#         Args:
#             None

#         Raises:
#             None

#         Returns:
#             None
#         """

#         total_rows = len(self.merged_df)

#         # remove target_variable. Do not impute target variable
#         non_target_columns = self.merged_df.drop(columns=targe_variable, axis=1).columns

#         # Iterate through columns to check for missing values
#         for column in non_target_columns:
#             missing_values = self.merged_df[column].isnull().sum()
#             missing_percentage = (missing_values / total_rows) * 100

#             # Drop columns with more than 50% missing values
#             if missing_percentage > conf_params["missing_percentage_threshold"]:
#                 logging.info(
#                     f"Dropping column '{column}' due to high percentage ({missing_percentage:.2f}%) of missing values."
#                 )
#                 self.merged_df.drop(columns=column, inplace=True)

#             # Impute missing values for columns with less than 50% missing values
#             elif missing_percentage > 0:
#                 logging.info(
#                     f"Column '{column}' has {missing_percentage:.2f}% missing values. Imputing..."
#                 )
#                 self.impute_null_data(column)

#         return None

#     def impute_null_data(self, column: str) -> None:
#         """Imputes missing values in the specified column.

#         Median is used for numeric columns and mode for non-numeric columns.

#         Args:
#             column (str): Name of the column to impute.

#         Raises:
#             None

#         Returns:
#             None
#         """
#         # Ensure that the column argument is of type str
#         if not isinstance(column, str):
#             logging.debug(
#                 "column argument is not of str type, will be typecasted to str."
#             )
#             column = str(column)

#         # Check if the column exists in the DataFrame
#         if column not in self.merged_df.columns:
#             logging.error(f"Column '{column}' does not exist in the DataFrame.")
#             return None

#         if np.issubdtype(self.merged_df[column].dtype, np.number):
#             median_value = self.merged_df[column].median()
#             self.merged_df[column].fillna(median_value, inplace=True)
#             logging.info(
#                 f"Imputed missing values in column '{column}' with median value of {median_value}.{self.merged_df[column].dtype}"
#             )
#         else:
#             mode_value = self.merged_df[column].mode()[0]
#             self.merged_df[column].fillna(mode_value, inplace=True)
#             logging.info(
#                 f"Imputed missing values in column '{column}' with mode value of {mode_value}.{self.merged_df[column].dtype}"
#             )

#         return None

#     def check_data_types(self) -> None:
#         """Checks and casts the data types of columns based on the expected data types provided.

#         Args:
#             None

#         Raises:
#             Exception: When type casting of dataframe columns is not possible with specified data type.

#         Returns:
#             None
#         """

#         # Loop through each column in DataFrame
#         for i, column in enumerate(self.merged_df.columns):
#             # Log the name of the current column being processed
#             # logging.info(f"{i+1}. Column: '{column}'.")

#             # Check if column is in the expected data types dictionary
#             if column in expected_dtypes.keys():
#                 expected_dtype = expected_dtypes[column]
#                 actual_dtype = self.merged_df[column].dtype.name

#                 # Check if the actual data type matches the expected data type and log.
#                 if actual_dtype != expected_dtype:
#                     logging.info(
#                         f"{i+1}.Attempting to fix column '{column}' from {actual_dtype} to the expected {expected_dtype} data type."
#                     )

#                     # Type Cast the column to the expected data type with errors ignored which means no change
#                     self.merged_df[column] = self.merged_df[column].astype(
#                         dtype=expected_dtype, errors="ignore"
#                     )
#                     # Log info after successful casting
#                     logging.info(
#                         f"{i+1}. Column '{column}' is now of datatype {self.merged_df[column].dtype.name}."
#                     )
#                 else:
#                     # Log that the data type of the column is as expected
#                     logging.info(f"{i+1}. Column '{column}' has the correct data type.")
#             else:
#                 # Log a warning if the column was not found in the expected data types dictionary
#                 logging.info(
#                     f"Column '{column}' not found in the config file under expected_dtypes. Leaving the datatype as it is."
#                 )

#         return None

#     def check_and_handle_data_constraints(self) -> None:
#         """Ensures data in certain columns meets specified constraints.

#         Args:
#             None

#         Raises:
#             None

#         Returns:
#             None
#         """
#         # Check if any of the columns have negative values
#         numeric_columns = self.merged_df.select_dtypes(include=["int", "float"]).columns
#         negative_columns = self.merged_df[numeric_columns].columns[
#             (self.merged_df[numeric_columns] < 0).any()
#         ]
#         # If there are negative_columns, log it.
#         if list(negative_columns):
#             logging.info(
#                 f"Columns with negative values: '{list(negative_columns)}. Please consider adding columns to non_negative_exogeneous_columns"
#             )

#         for i, column in enumerate(self.merged_df.columns):
#             # Ensuring non-negative values
#             if column in non_negative_exogeneous_columns:
#                 # Set lower bound of range to 0
#                 self.merged_df[column] = self.merged_df[column].clip(lower=0)
#                 logging.info(f"Ensured non-negative values in column '{column}'.")

#             # Ensuring minimum value of 1 for 'propensity_factor' column
#             elif column == "propensity_factor":
#                 self.merged_df[column] = self.merged_df[column].clip(lower=1)
#                 logging.info(f"Ensured minimum value of 1 in column '{column}'.")

#             # Log that data constraints have been successfully handled.
#             # logging.info(f" Column {i}: {column} have been successfully handled.")

#         return None
