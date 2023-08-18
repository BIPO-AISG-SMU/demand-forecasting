from bipo.pipelines.data_loader.data_check import DataCheck
import pandas as pd
import sys
from kedro.config import ConfigLoader
from kedro.framework.project import settings

from bipo.pipelines.data_loader.dataloader import check_outlet_location
from bipo.utils import get_project_path

# initiate to constants.yml and parameters.yml
project_path = get_project_path()
conf_loader = ConfigLoader(conf_source=project_path / "conf")
conf_params = conf_loader["parameters"]["dataloader"]  # use this if got paramaters
constants = conf_loader.get("constants*")["dataloader"]

EXPECTED_COLUMNS = constants["expected_columns"]["propensity"]
NEW_COLUMN_NAMES = constants["new_column_names"]["propensity"]


class ReadPropensityData(DataCheck):
    """reads consumer propensity to spend data into a dataframe.

    Inherits the data_check class.
    """

    def __init__(self, proxy_revenue_df):
        super().__init__()
        self.proxy_revenue_df = proxy_revenue_df

    def read_data(self, file_path: str) -> pd.DataFrame:
        """Function to read, check file format encoding, check if data exist and check for expected columns and set time index for consumer propensity data

        Args:
            file_path (str): path to propensity data

        Returns:
            propensity_df (pandas.core.frame.DataFrame): propensity dataframe
        """
        # Check if file is in the correct format encoding
        self.check_file_format_encoding(file_path)

        propensity_df = pd.read_excel(file_path, engine="openpyxl")

        # Check data exists
        self.check_data_exists(propensity_df)

        # Set date as index before checking for columns
        propensity_df.set_index("Date", inplace=True)

        # Check if expected columns exist
        self.check_columns(propensity_df.columns.tolist(), "propensity")

        return propensity_df

    def preprocess_data(self, propensity_df, location: str) -> pd.DataFrame:
        """Get the location of the specific cost centre and filter propensity_df based on specific location. Return the location specific propensity_df

        Args:
            propensity_df (pandas.core.frame.DataFrame): Input from read_data

        Returns:
            propensity_df (pandas.core.frame.DataFrame): location specific propensity_df
        """
        # create a copy
        processed_propensity_df = propensity_df.copy()

        processed_propensity_df = processed_propensity_df[
            processed_propensity_df["Location"] == location
        ]

        # Drop the "Location" column
        processed_propensity_df = processed_propensity_df.drop("Location", axis=1)

        # change the column names according to data_loader_validation.yml
        processed_propensity_df = processed_propensity_df.rename(
            columns=dict(zip(EXPECTED_COLUMNS, NEW_COLUMN_NAMES))
        )
        return processed_propensity_df
