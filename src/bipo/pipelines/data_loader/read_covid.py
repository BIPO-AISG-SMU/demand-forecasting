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

EXPECTED_COLUMNS = constants["expected_columns"]["covid_record"]
NEW_COLUMN_NAMES = constants["new_column_names"]["covid_record"]


class ReadCovidData(DataCheck):
    """reads and preprocess covid data into a dataframe.

    Inherits the DataCheck class.
    """

    def __init__(self):
        super().__init__()

    def read_data(self, file_path: str) -> pd.DataFrame:
        """Read covid data, check for file format encoding, check for empty dataframe, check for expected columns and set datatime index.

        Args:
            file_path (pandas.core.frame.DataFrame): file path to covid data

        Returns:
            covid_df (pandas.core.frame.DataFrame): covid dataframe
        """

        # Check if file is in the correct format encoding
        self.check_file_format_encoding(file_path)

        covid_df = pd.read_excel(file_path, sheet_name="Sheet2", engine="openpyxl")

        covid_df = covid_df.set_index("Date")
        self.check_data_exists(covid_df)
        self.check_columns(covid_df.columns.tolist(), "covid_record")
        return covid_df

    def preprocess_data(self, covid_df: pd.DataFrame):
        """Function change column names of covid_df.

        Args:
            covid_df (pandas.core.frame.DataFrame): input from read_data

        Returns:
            preprocessed_covid_df (pandas.core.frame.DataFrame): preprocessed covid dataframe
        """
        # change the column names
        preprocessed_covid_df = covid_df.copy()
        column_mapping = dict(zip(EXPECTED_COLUMNS, NEW_COLUMN_NAMES))
        preprocessed_covid_df = preprocessed_covid_df.rename(columns=column_mapping)
        return preprocessed_covid_df
