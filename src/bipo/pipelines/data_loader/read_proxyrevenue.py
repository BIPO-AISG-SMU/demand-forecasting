from bipo.pipelines.data_loader.data_check import DataCheck
import logging
import pandas as pd
import sys
from kedro.config import ConfigLoader
from kedro.framework.project import settings
from bipo.utils import get_project_path  # , get_logger

# logging = logging.getLogger("kedro")
logging = logging.getLogger(__name__)

# initiate to constants.yml and parameters.yml
project_path = get_project_path()
conf_loader = ConfigLoader(conf_source=project_path / "conf")
conf_params = conf_loader["parameters"]["dataloader"]
constants = conf_loader.get("constants*")["dataloader"]

EXPECTED_COLUMNS = constants["expected_columns"]["proxy revenue"]
NEW_COLUMN_NAMES = constants["new_column_names"]["proxy revenue"]

LOCATION_RENAME_MAPPING = conf_params["revenue_location_rename_mapping"]


class ReadProxyRevenueData(DataCheck):
    """reads Proxy Revenue data into a dataframe.

    Inherits the DataCheck class.
    """

    def __init__(self):
        super().__init__()

    def read_data(self, file_path: str) -> pd.DataFrame:
        """Read Proxy Revenue data, check for file format encoding, check for empty dataframe , check for expected columns and remove cost centre that do not have at least 2 years data. Change the Location names to common names used in other data. eg Propensity data

        Args:
            file_path (str): file path to Proxy Revenue data

        Returns:
            proxy_revenue_df (pd.DataFrame): proxy df with only cost centre with 2 years or more data.
        """
        # Check if file is in the correct format encoding
        self.check_file_format_encoding(file_path)

        # read data
        proxy_revenue_df = pd.read_csv(file_path)
        self.check_data_exists(proxy_revenue_df)
        # check for columns
        self.check_columns(proxy_revenue_df.columns.tolist(), "proxy revenue")
        # Get the count of data for each CostCentreCode
        outlet_data_counts = proxy_revenue_df["CostCentreCode"].value_counts()

        # Get the CostCentreCodes with more than or equal to 2 years of data
        threshold_days = conf_params["number_of_days"]
        valid_codes = outlet_data_counts[
            outlet_data_counts >= threshold_days
        ].index.tolist()
        non_valid_codes = outlet_data_counts[
            outlet_data_counts < threshold_days
        ].index.tolist()
        # Have a log here to record how many costcenter dropped
        logging.info(
            f"Cost center with less than {threshold_days} days of data: {non_valid_codes}, length{len(non_valid_codes)}"
        )

        # Filter the DataFrame to include only valid CostCentreCodes
        proxy_revenue_df = proxy_revenue_df[
            proxy_revenue_df["CostCentreCode"].isin(valid_codes)
        ]
        # Change the location names. These names will be used for mapping
        proxy_revenue_df["Location"] = proxy_revenue_df["Location"].replace(
            LOCATION_RENAME_MAPPING
        )
        # Change int to str so that kedro can read it
        proxy_revenue_df["CostCentreCode"] = proxy_revenue_df["CostCentreCode"].astype(
            str
        )
        self.proxy_revenue_df = proxy_revenue_df
        # Have to return proxy_revenue_df as it will be used for location mapping

        return proxy_revenue_df

    def preprocess_data(self, proxy_revenue_df, cost_centre_code: str) -> pd.DataFrame:
        """Filter for specific cost centre and return a cost centre specific Proxy Revenue dataframe.

        Args:
            proxy_revenue_df (pd.DataFrame): output from read_data is served as the input for this function
            cost_centre_code (str): Specific cost center code of interest.

        Returns:
            preprocess_proxy_revenue_df (pd.DataFrame): Cost Centre Specific Proxy Revenue Dataframe
        """
        # # filter for specific outlet. This is inside the pipeline loop
        preprocess_proxy_revenue_df = proxy_revenue_df.copy()

        # Filter for the specific outlet using boolean indexing
        preprocess_proxy_revenue_df = preprocess_proxy_revenue_df[
            preprocess_proxy_revenue_df["CostCentreCode"] == cost_centre_code
        ]

        # Convert the 'Date' column to datetime and set it as the DataFrame index in one step
        preprocess_proxy_revenue_df["Date"] = pd.to_datetime(
            preprocess_proxy_revenue_df["Date"]
        )
        preprocess_proxy_revenue_df.set_index("Date", inplace=True)

        # change the column names according to data_loader_validation.yml
        column_mapping = dict(
            zip(
                EXPECTED_COLUMNS,
                NEW_COLUMN_NAMES,
            )
        )
        preprocess_proxy_revenue_df = preprocess_proxy_revenue_df.rename(
            columns=column_mapping
        )

        return preprocess_proxy_revenue_df
