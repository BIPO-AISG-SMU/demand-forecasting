# Python script that contains series of functions that can be used for python scripts in feature_engineering
from typing import Union, List, Tuple, Dict
import pandas as pd
import numpy as np
import sys

sys.dont_write_bytecode = True
import logging

logging = logging.getLogger(__name__)


def apply_equal_width_binning(
    df: pd.DataFrame,
    column_name: str,
    bin_column_name: str,
    bin_labels_list: list,
) -> Union[None, pd.DataFrame]:
    """This function applies an equal width binning on the class\
        instance's dataframe's column of interest by using pandas'\
        cut method. Number of bins is dependent on the length of bin_labels_list specified. 

    Args:
        df (pd.DataFrame): Dataframe to be processed.
        column_name (str): Column name of a pd.DataFrame.
        bin_column_name (str): Column name to be created for\
            storing binned values.
        bin_labels_list (list): List containing bin labels

    Raises:
        KeyError: When column argument passed is not found in the\
            dataframe columns.
        ValueError: If length of labels_list is not 4 or the values\
            in the column of dataframe are non-numerical.

    Returns:
        None: if non-dataframe was passed or KeyError or ValueError is encountered.
        pd.DataFrame: Dataframe with added column representing binned feature.
    """
    if not isinstance(df, pd.DataFrame):
        log_string = "df argument is not of pd.DataFrame type. Will not return anything as a result."
        logging.error(log_string, stack_info=True)
        return None

    if not isinstance(column_name, str):
        log_string = "column argument is not of string type, will be typecasted to str"
        logging.debug(log_string)
        column_name = str(column_name)

    if not isinstance(bin_column_name, str):
        log_string = (
            "bin_column_name argument is not of string type, will be typecasted to str"
        )
        logging.debug(log_string)
        bin_column_name = str(bin_column_name)

    if not isinstance(bin_labels_list, list):
        log_string = (
            "bin_labels_list argument is not of list type, will be typecasted to list"
        )
        logging.debug(log_string)
        bin_labels_list = list(str(bin_labels_list))
        logging.debug(f"Generated list: {bin_labels_list}")

    try:
        df[bin_column_name] = pd.cut(
            df[column_name],
            bins=len(bin_labels_list),
            labels=bin_labels_list,
            right=False,
            include_lowest=True,
        )
        return df[[bin_column_name]]

    except KeyError:
        log_string = f"The column {column_name} is not in the dataframe to be\
            processed"
        logging.error(log_string, stack_info=True)

        return None

    except ValueError:
        log_string = f"The column {column_name} is not numerical hence the requested binning could not executed."
        logging.error(log_string, stack_info=True)

        return None


# Number of bins use would need to confirm with PS
def apply_mean_std_binning(
    df: pd.DataFrame,
    column_name: str,
    bin_column_name: str,
    bin_labels_list: list = ["1", "2", "3", "4"],
) -> Union[None, pd.DataFrame]:
    """This function constructs a new dataframe column representing
    one of the four bin categories to be constructed, with 
    information based on labels_list argument. 
    
    The four bin edges are defined with the following intervals: \
    [min, mean - std), [mean-SD,mean), [mean, mean+std) and
    [mean+1.5*std,max). The bin labels will be set as 1, 2, 3 and 4 by default. 


    Args:
        df (pd.DataFrame): Dataframe to be processed.
        column_name (str): Column name of a pd.dataframe.
        bin_column_name (str): Column name to be created for\
            storing binned values.

    Raises:
        TypeError: When input arguments do not match the expected\
            types.
        KeyError: When feature argument passed is not found in the\
            dataframe columns.
        ValueError: If length of labels_list is not 4 or the values\
            in the feature of dataframe are non-numerical.

    Returns:
        None: if non-dataframe was passed.
        pd.DataFrame: Dataframe containing an additional column representing column that was binned if successful. Otherwise, no change is made.

    """
    if not isinstance(df, pd.DataFrame):
        log_string = "df argument is not of pd.DataFrame type. Will not return anything as a result"
        logging.error(log_string, stack_info=True)
        return None

    if not isinstance(bin_labels_list, list):
        log_string = (
            "bin_labels_list argument is not list type. Will use default labels"
        )
        logging.error(log_string, stack_info=True)
        bin_labels_list = ["1", "2", "3", "4"]

    if not isinstance(column_name, str):
        log_string = "column argument is not of string type, will be typecasted to str"
        logging.debug(log_string, stack_info=True)
        column_name = str(column_name)

    if not isinstance(bin_column_name, str):
        log_string = (
            "bin_column_name argument is not of string type, will be typecasted to str"
        )
        logging.debug(log_string, stack_info=True)
        bin_column_name = str(bin_column_name)

    # Check the bin_labels_list is it 4
    if len(bin_labels_list) != 4:
        bin_labels_list = ["Low", "Medium", "High", "Very High"]

    try:
        # Derive value of bin edges requires min, max, mean and std values
        min_val = df[column_name].min()
        max_val = df[column_name].max()
        mean_val = df[column_name].mean()
        std_val = df[column_name].std()

        # General statistics
        log_string = f"Generated stats - Min: {min_val}, Max: {max_val}, Mean: {mean_val}, Std: {std_val}"
        logging.info(log_string)

        # Construct binning using bin edge values. Total 4 bins to be created. with min and max edges set to 0 and infinity.
        bin_edges_list = [
            0,
            mean_val - std_val,
            mean_val,
            mean_val + std_val,
            np.inf,
        ]

        logging.debug(
            f"Applying pd cut on {column_name}... using bin edges: {bin_edges_list}"
        )
        # Construct 4 bins based on bin_edges_list
        df[bin_column_name] = pd.cut(
            df[column_name],
            bins=bin_edges_list,
            labels=bin_labels_list,
            right=False,
            include_lowest=True,
        )

        return df[[bin_column_name]]
    except KeyError:
        log_string = f"{column_name} is not in the dataframe to be\
            processed"
        logging.error(log_string, stack_info=True)

        return None

    except ValueError:
        log_string = f"{column_name} is not numerical\
                hence the requested binning could not executed."
        logging.error(log_string, stack_info=True)

        return None


def apply_equal_freq_binning(
    df: pd.DataFrame,
    column_name: str,
    bin_column_name: str,
    bin_labels_list: list,
) -> Union[None, pd.DataFrame]:
    """Function that creates equal frequency binning by applying pandas qcut approach to divide the data so that the number of elements in each bin is as equal as possible on the specified column.

    Args:
        df (pd.DataFrame): Dataframe to be processed.
        column (str): Column name of a pd.dataframe.
        bin_column_name(str):\
            Column name to be created for\
            storing binned values.
        bin_labels_list (list):\
            Bin labels used for binning approach. Defaults to []

    Returns:
        None: if non-dataframe was passed.
        pd.DataFrame: Dataframe with added column representing binned feature.

    """
    if not isinstance(df, pd.DataFrame):
        log_string = "df argument is not of pd.DataFrame type. Will not return anything as a result."
        logging.error(log_string, stack_info=True)
        return None

    if not isinstance(column_name, str):
        log_string = "column argument is not of string type"
        logging.debug(log_string, stack_info=True)
        column_name = str(column_name)

    if not isinstance(bin_column_name, str):
        log_string = "bin_column_name argument is not of string type"
        logging.debug(log_string, stack_info=True)
        bin_column_name = str(bin_column_name)

    if not isinstance(bin_labels_list, list):
        log_string = (
            "bin_labels_list argument is not of list type, will be typecasted to list"
        )
        logging.debug(log_string)
        bin_labels_list = list(str(bin_labels_list))
        logging.debug(f"Generated list: {bin_labels_list}")

    try:
        df[bin_column_name] = pd.qcut(
            df[column_name], q=len(bin_labels_list), labels=bin_labels_list
        )

        return df[[bin_column_name]]
    except KeyError:
        log_string = f"Attempt to access {column_name}, but was not in the      dataframe to be processed. Return as it is."
        logging.error(log_string, stack_info=True)

        return None

    except ValueError:
        log_string = f"{column_name} is not numerical hence the requested binning could not executed. Return as it is."
        logging.error(log_string, stack_info=True)

        return None


# def bin_numerical_features(
#     df,
#     column_name: str,
#     bin_approach: str,
#     bin_edges_list: list = [],
#     bin_labels_list: list = [],
# ):
#     """Function creates a new dataframe column representing the
#         binned numerical features in a dataframe based on provided bin
#         edges values and their corresponding labels.

#         If column contains 'total' substring, binning will be made based
#         on the SALES_BIN_APPROACH global variable that dictates the binning required for .

#         Left edge values of defined bins are included as part of cut.

#         Args:
#             df (pd.DataFrame): Pandas dataframe to be binned.
#             column_name (str): Column name of a dataframe.
#             bin_column_name (str): Column name to be created for\
#                 storing binned values.
#             bin_approach (str): Bin approach specified in string.
#             bin_edges_list(list):\
#                 Bin boundaries used for binning approach. Defaults to []
#             bin_labels_list (list):\
#                 Bin labels used for binning approach. Defaults to []

#         Raises:
#             TypeError: When input arguments do not match the expected\
#                 types.
#             KeyError: When feature argument passed is not found in the\
#                 dataframe columns.
#             ValueError: If length of labels_list is not 4 or the values\
#                 in the feature of dataframe are non-numerical.

#         Returns:
#             None
#         """

#     if not isinstance(column_name, str):
#         log_string = "column argument is not of string type, will be typecasted to str."
#         logging.debug(log_string)
#         column_name = str(column_name)

#     if not isinstance(bin_approach, str):
#         log_string = (
#             "bin_approach argument is not of string type, will be typecasted to str."
#         )
#         logging.debug(log_string)
#         bin_approach = str(bin_approach)

#     if not isinstance(bin_edges_list, list):
#         log_string = (
#             "bin_edges_list argument is not of string type, will be typecasted to list."
#         )
#         logging.error(log_string)
#         bin_edges_list = list(bin_edges_list)

#     if not isinstance(bin_labels_list, list):
#         log_string = "bin_labels_list argument is not of string type, will be typecasted to list."
#         logging.debug(log_string)
#         bin_edges_list = list(bin_labels_list)

#     # Create merged_df for binned numerical features

#     # Create column name for column to be binned. Eg for total_tc creates
#     bin_column_name = "cat_" + column_name

#     logging.info(f"Creating bins for {column_name}")

#     # Covers total_sales
#     if column_name.lower() == "total_sales":
#         # Various Binning approaches
#         if bin_approach == "Equal Width Binning":
#             # Temporary
#             bin_column_name = "equal_width_bin" + column_name

#             df = apply_equal_width_binning(
#                 df=df,
#                 column=column_name,
#                 bin_column_name=bin_column_name,
#                 bin_labels_list=bin_labels_list,
#                 logging=logging,
#             )

#         elif bin_approach == "Mean Std Binning":
#             # Temporary
#             bin_column_name = "mean_std_bin" + column_name
#             df = apply_mean_std_binning(
#                 df=df,
#                 column=column_name,
#                 bin_column_name=bin_column_name,
#                 bin_labels_list=bin_labels_list,
#                 logging=logging,
#             )

#         elif bin_approach == "Equal Frequency Binning":
#             # Temporary
#             bin_column_name = "eq_freq_bin" + column_name
#             df = apply_equal_freq_binning(
#                 df=df,
#                 column=column,
#                 bin_column_name=bin_column_name,
#                 bin_labels_list=bin_labels_list,
#                 logging=logging,
#             )

#         else:
#             log_string = f"Using default equal width binning due to invalid bin approach specified."
#             logging.info(log_string, stack_info=True)

#             df = apply_equal_width_binning(
#                 df=df,
#                 column=column_name,
#                 bin_column_name=bin_column_name,
#                 bin_labels_list=bin_labels_list,
#                 logging=logging,
#             )

#         logging.info(f"{column_name} has been binned.")

#         return df

#     # For other features
#     else:
#         try:
#             df[bin_column_name] = pd.cut(
#                 x=df[column_name],
#                 bins=bin_edges_list,
#                 labels=bin_labels_list,
#                 include_lowest=True,
#             )
#         except KeyError:
#             log_string = f"{column_name} is not in dataframe columns"
#             logging.error(log_string, stack_info=True)

#         except ValueError:
#             log_string = f"{column_name} contains non-numeric values"
#             logging.error(log_string, stack_info=True)

#         return df


def create_min_max_feature_diff(
    df,
    column_name_min: str,
    column_name_max: str,
) -> Union[None, pd.DataFrame]:
    """Function that calculates the difference between the 2 provided column of interest (column_min and column_max) of the dataframe.

        Args:
            df (pd.DataFrame): Pandas DataFrame of interest
            column_min (str): Column with minimum value of a feature.
            column_max (str): Column with maximum value of a feature.
            new_feature_name (str): Feature name that represents the difference value.

        Raises:
            KeyError: When either column_min or column_max argument \
                passed is not found in the dataframe columns.
            ValueError: If the values of the dataframe column concerned\
                is not numeric.

        Returns:
            None if df argument is not pandas DataFrame or KeyError or valueError is encountered. Otherwise pd.Series containing created feature representing the difference value of 2 columns specified.
        """

    if not isinstance(df, pd.DataFrame):
        log_string = "df argument is not of pandas DataFrame."
        logging.error(log_string)
        return None

    if not isinstance(column_name_min, str):
        log_string = (
            "column_min argument is not of str type, will be typecasted to str."
        )
        logging.debug(log_string)
        column_name_min = str(column_name_min)

    if not isinstance(column_name_max, str):
        log_string = (
            "column_max argument is not of str type, will be typecasted to str."
        )
        logging.debug(log_string)
        column_name_max = str(column_name_max)

    logging.info(
        f"Calculating the difference between {column_name_min} & {column_name_max} features."
    )

    try:
        # Create new feature name to represent the max-min value as return value
        df = df.copy()
        df["diff"] = df[column_name_max] - df[column_name_min]

        return df[["diff"]]

    except KeyError:
        log_string = f"Unable to read either columns {column_name_min} or {column_name_max} as either does not exist in the dataframe."
        logging.error(log_string, stack_info=True)

        return None

    except ValueError:
        log_string = f"Unable to execute numeric operation provided column {column_name_min} or {column_name_max} as it not either a float/int column."
        logging.error(log_string, stack_info=True)

        return None


def read_csv_file(filepath: str) -> Union[None, pd.DataFrame]:
    """Function that reads in a file located in the specified filepath into a pandas DataFrame.

    Args:
        filepath (str): Absolute filepath of file to be read.

    Raises:
        IOError: When limited permission involving file reading is encountered.
        FileNotFoundError: If the path specified does not exists.

    Returns:
        None
    """
    if not isinstance(filepath, str):
        log_string = "filepath argument is not of str type, will be typecasted to str."
        logging.debug(log_string)
        filepath = str(filepath)

    try:
        df = pd.read_csv(filepath, index_col=0)
        return df

    except IOError:
        log_string = f"Unable to read file in {filepath}. Please ensure necessary permissions have been granted"
        logging.error(log_string, stack_info=True)

        return None

    except FileNotFoundError:
        log_string = (
            f"Unable to locate file in {filepath}. Please check that it exists."
        )
        logging.error(log_string, stack_info=True)

        return None

    return None


def get_group_number(
    df: pd.DataFrame,
    days_per_group: int = 7,
    new_column_name: str = "group",
) -> Union[None, pd.DataFrame]:
    """Helper function called by generate_lag_avg_weekly_sales to generate group information for each date entry of the dataframe based on number of entries (days_per_group) to form as a group and the start date which grouping (start_day) should start.

    Example, you can group starting every monday (based on start_day="Monday" for a specified period (days_per_group). Every 7 days could be a group and the dataset will be split by weeks. Adds an additional 'group' column for the group number to the original dataframe.

    Args:
        days_per_group (int, optional): Number of days per group. Default is 7 days.
        new_column_name (str): Column name of dataframe instance containing grouping information.

    Returns:
        pd.DataFrame: Subset dataframe containing added group column. None if non-dataframe is passed or KeyError is encountered.
    """

    if not isinstance(df, pd.DataFrame):
        log_string = "df argument is not of pd.DataFrame type, will skip process and return None."
        logging.info(log_string)
        return None

    if not isinstance(days_per_group, int):
        log_string = "days_per_group argument is not of int type, will be typecasted set to default value 7 (a week)."
        logging.info(log_string)
        days_per_group = 7

    if not isinstance(new_column_name, str):
        log_string = (
            "new_column_name argument is not of str type, will be set to value 'group'."
        )
        logging.info(log_string)
        new_column_name = "group"

    start_index = df.index[0]
    # Create filtered subset dataframe based on start index
    subset_df = df.loc[start_index:, :].copy()

    # Group values will start from index 1 onwards. Eg 1,1,1....,2,2....
    subset_df[new_column_name] = (
        np.arange(len(df.loc[start_index:])) // days_per_group + 1
    )

    logging.debug("Generated data groups via helper function")
    return subset_df
