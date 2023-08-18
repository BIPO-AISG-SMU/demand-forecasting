from bipo.pipelines.data_loader.data_check import DataCheck
import pandas as pd
import sys
from kedro.config import ConfigLoader
from kedro.framework.project import settings

from bipo.pipelines.data_loader.dataloader import check_outlet_location

# from ...utils import get_project_path
from bipo.utils import get_project_path

# initiate to constants.yml and parameters.yml
project_path = get_project_path()
conf_loader = ConfigLoader(conf_source=project_path / "conf")
conf_params = conf_loader["parameters"]["dataloader"]  # use this if got paramaters
constants = conf_loader.get("constants*")["dataloader"]

EXPECTED_COLUMNS = constants["expected_columns"]["climate"]
NEW_COLUMN_NAMES = constants["new_column_names"]["climate"]


class ReadClimateData(DataCheck):
    """reads climate data into a dataframe.

    Inherits the DataCheck class.
    """

    def __init__(self, proxy_revenue_df: pd.DataFrame):
        super().__init__()
        self.proxy_revenue_df = proxy_revenue_df

    def read_data(self, file_path: str) -> pd.DataFrame:
        """Read Climate data, check for file format encoding, check for empty dataframe , check for expected columns.

        Args:
            file_path (str): File path to climate data

        Returns:
            climate_df (pandas.core.frame.DataFrame): climate dataframe
        """

        # Check if file is in the correct format encoding
        self.check_file_format_encoding(file_path)

        climate_df = pd.read_excel(file_path, engine="openpyxl")

        # Check data exists
        self.check_data_exists(climate_df)

        # Check columns
        self.check_columns(climate_df.columns.tolist(), "climate")

        return climate_df

    def preprocess_data(self, climate_df, location: str) -> pd.DataFrame:
        """Function to convert "Year", "Month", "Day" to datetime index, get cost centre location and filter climate_df for the specific location information.

        Args:
            climate_df (pd.DataFrame): Input from read_data

        Returns:
            preprocessed_climate_df(pd.DataFrame): preprocessed location specific climate df
        """
        preprocessed_climate_df = climate_df.copy()

        # Combine year, month, and day columns into a single "Date" column
        preprocessed_climate_df["Date"] = pd.to_datetime(
            preprocessed_climate_df[["Year", "Month", "Day"]]
        )

        # Filter by location
        preprocessed_climate_df = preprocessed_climate_df[
            preprocessed_climate_df["DIRECTION"] == location
        ]
        preprocessed_climate_df = preprocessed_climate_df.set_index("Date")

        # Drop unnecessary columns
        preprocessed_climate_df.drop(
            columns=["DIRECTION", "Year", "Month", "Day"], inplace=True
        )

        # Rename columns
        column_mapping = dict(zip(EXPECTED_COLUMNS, NEW_COLUMN_NAMES))
        preprocessed_climate_df = preprocessed_climate_df.rename(columns=column_mapping)

        return preprocessed_climate_df
