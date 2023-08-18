from bipo.pipelines.data_loader.data_check import DataCheck
import pandas as pd
import sys
from kedro.config import ConfigLoader
from kedro.framework.project import settings
from bipo.utils import get_project_path


# initiate to constants.yml and parameters.yml
project_path = get_project_path()
conf_loader = ConfigLoader(conf_source=project_path / "conf")
conf_params = conf_loader["parameters"]["dataloader"]  # use this if got paramaters
constants = conf_loader.get("constants*")["dataloader"]

EXPECTED_COLUMNS = constants["expected_columns"]["holiday"]
NEW_COLUMN_NAMES = constants["new_column_names"]["holiday"]


class ReadHolidayData(DataCheck):
    """reads and preprocess holiday data into a dataframe.

    Inherits the DataCheck class.
    """

    def __init__(self):
        super().__init__()

    def read_data(self, file_path: str) -> pd.DataFrame:
        """Function to read, check file format encoding, check if data exist and check for expected columns for holiday data.

        Args:
            file_path (str): File Path to holiday data

        Returns:
            holiday_df (pandas.core.frame.DataFrame): holiday dataframe
        """
        # Check if file is in the correct format encoding
        self.check_file_format_encoding(file_path)
        holiday_df = pd.read_excel(file_path, engine="openpyxl")
        self.check_data_exists(holiday_df)
        self.check_columns(holiday_df.columns.tolist(), "holiday")
        return holiday_df

    def preprocess_data(self, holiday_df: pd.DataFrame) -> pd.DataFrame:
        """Function to set time index, drop useless columns, convert the School and Public holiday to True or False and change the column names.

        Args:
            holiday_df (pandas.core.frame.DataFrame): Input from read_data

        Returns:
            preprocessed_holiday_df (pandas.core.frame.DataFrame): preprocessed holiday dataframe
        """
        preprocessed_holiday_df = holiday_df.copy()
        preprocessed_holiday_df["Date"] = pd.to_datetime(
            preprocessed_holiday_df["Date"]
        )
        preprocessed_holiday_df = preprocessed_holiday_df.set_index("Date")

        # drop useless columns
        preprocessed_holiday_df.drop(
            ["School Holiday Type", "Public Holiday Type"], axis=1, inplace=True
        )

        # Convert to Yes or No
        preprocessed_holiday_df["School Holiday"] = preprocessed_holiday_df[
            "School Holiday"
        ].apply(lambda x: True if x == "Yes" else False)
        preprocessed_holiday_df["Public Holiday"] = preprocessed_holiday_df[
            "Public Holiday"
        ].apply(lambda x: True if x == "Yes" else False)

        # change the column names
        column_mapping = dict(zip(EXPECTED_COLUMNS, NEW_COLUMN_NAMES))
        preprocessed_holiday_df = preprocessed_holiday_df.rename(columns=column_mapping)

        return preprocessed_holiday_df
