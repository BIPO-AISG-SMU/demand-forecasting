from typing import List, Optional, Tuple, Union
import pandas as pd
import os
from kedro.config import ConfigLoader
from .common import create_min_max_feature_diff
from bipo.utils import get_logger, get_project_path
import logging

logging = logging.getLogger(__name__)
# Instantiate config
project_path = get_project_path()
conf_loader = ConfigLoader(conf_source=project_path / "conf")
conf_params = conf_loader["parameters"]
constants = conf_loader.get("constants*")

RAINFALL_THRESHOLD = constants["rain_threshold"]
RAINFALL_COLUMN = constants["rainfall_column"]


class Exogenous:
    """A class used to perform feature engineering involving exogenous features of a dataset.

    This class has methods for extracting and transforming exogenous features.

    Attributes:
        filepath (str): Path to file that is to be read.
        added_features_exog (List): List of added exogenous features.
        exog_df (pd.Dataframe): Dataframe obtained from reading csv file. Otherwise empty if filepath does not exist.
    """

    def __init__(self, outlet_df: pd.DataFrame) -> None:
        """Initializes the Exog class.

        Args:
            outlet_df (pd.DataFrame): Individual outlet dataframe
        """
        self.added_features_exog = []
        self.exog_df = outlet_df

        return None

    def get_added_features_exog(self) -> list:
        """Getter function for added_feature_exog.

        Args:
            None

        Raises:
            None

        Returns:
            list: self.added_features_exog instance variable
        """
        return self.added_features_exog

    def exog_transform(self) -> Union[None, pd.DataFrame]:
        """Extracts and transforms exogenous features from the data frame.

        This function extracts exogenous features based on the configuration, applies transformations
        like binning, and concatenates the extracted features into a single DataFrame.

        Returns:
            pd.DataFrame: A DataFrame containing the extracted and transformed exogenous features.
        """

        # Create DataFrame for exogenous features not requiring binning
        logging.info("Subsetting features that require no binning")
        exog_nobin_df = self._create_exog_nobin_dataframe()
        logging.info(
            f"Finished subsetting features that require no binning with {exog_nobin_df.shape}"
        )
        # Create DataFrame for binned exogenous features
        logging.info("Creating binning features")
        # exog_bin_df = self._create_exog_bin_dataframe()
        # logging.info(f"Finished creating binning features with {exog_bin_df.shape}")

        # Add new feature for the difference between min and max temperatures. Changes is made directly to dataframe instance of the class.
        logging.info("Creating min max temperature difference feature")
        # exog_added_df = self._add_min_max_temp_diff_feature()
        # logging.info(
        #     f"Finished min max temperature difference feature with {exog_added_df.shape}"
        # )

        # generate feature is_raining
        rain_df = self.add_is_raining_feature(RAINFALL_COLUMN, RAINFALL_THRESHOLD)

        # Create a copy to start merging
        # exog_concat_df = exog_added_df.copy()
        exog_concat_df = self.exog_df.copy()
        exog_concat_df.drop(
            conf_params["feature_engineering"]["exog_feature_nobin"],
            axis=1,
            inplace=True,
        )

        # List of dataframe to concat
        df_merge_list = [exog_nobin_df, rain_df]
        log_string = "Concatenating dataframes containing binned/feature engineered exogenous variables."
        logging.debug(log_string)

        try:
            for df in df_merge_list:
                exog_concat_df = exog_concat_df.merge(
                    df, how="left", left_index=True, right_index=True
                )
        except ValueError:
            log_string = "Encountered errors when merging the dataframes involving exogenous variables. Please check."
            logging.error(log_string, exc_info=True)
            exog_concat_df = None

        logging.debug(f"Source dataframe contains {self.exog_df.shape[1]} features")

        logging.debug(
            f"Added total {len(self.added_features_exog)} features from feature engineering"
        )

        logging.debug(
            f"Concatenated exogenous dataframe shape: {exog_concat_df.shape} with columns: {exog_concat_df.columns}\n"
        )

        return exog_concat_df

    def _add_min_max_temp_diff_feature(self) -> Union[None, pd.DataFrame]:
        """Adds a new feature representing the difference between minimum and maximum temperatures.

        This function retrieves temperature information from the configuration file, calculates the difference between the minimum and maximum temperatures, and adds this as a new feature to the DataFrame.

        Created feature will be added to self.added_features_exog for traceability.

        Args:
            None

        Raises:
            None

        Returns:
            pd.DataFrame containing the temperature differenced values.
        """
        # Retrieve temperature information from the configuration
        temperature = conf_params["feature_engineering"]["exog_feature_todiff"][
            "temperature"
        ]

        # Extract min and max temperatures from config
        if len(temperature) > 1:
            temp_min = temperature[0]
            temp_max = temperature[1]

        else:
            temp_min = None
            temp_max = None

        # Create new feature for the difference between min and max temperatures if there is data. Else do nothing.
        new_feature_name = "diff_temp"

        if temp_min and temp_max:
            log_string = (
                "Creating temperature difference feature from max and min temperature"
            )

            logging.debug(log_string)

            temp_df = create_min_max_feature_diff(
                df=self.exog_df,
                column_name_min=temp_min,
                column_name_max=temp_max,
            )

            temp_df.rename(columns={"diff": new_feature_name}, inplace=True)

            log_string = (
                "Created temperature difference feature from max and min temperature"
            )
            logging.debug(log_string)

            # Update added features:
            self.added_features_exog.extend([new_feature_name])
            logging.debug(
                f"Generated dataframe: {temp_df.shape} with columns {temp_df.columns}"
            )

            return temp_df

        else:
            log_string = "Configuration for temperature variable in config is not a list of 2 elements. Please check. Differencing for temperature would be skipped."
            logging.error(log_string, exc_info=True)

            return None

    def _create_exog_nobin_dataframe(self) -> Union[None, pd.DataFrame]:
        """Creates a DataFrame for exogenous features not requiring binning.

        Retrieves a list of exogenous features from the configuration file that do not require binning and extracts them into a new DataFrame. If a feature is not found in the DataFrame, it is skipped.

        Args:
            None

        Raises:
            None

        Returns:
            pd.DataFrame: A DataFrame containing only the exogenous features that do not require binning if config specification is non-empty. Else, return None.
        """
        # Retrieve exog_feature_nobin information from the configuration
        exog_feature_nobin_list = conf_params["feature_engineering"][
            "exog_feature_nobin"
        ]

        # Log a warning if no exogenous features are found in the configuration file and returns empty dataframe.
        if not exog_feature_nobin_list:
            logging.warning(
                "No exogenous features found in the configuration for direct inclusion."
            )
            return None

        logging.info(
            f"Creating exogenous features dataframe that do not need binning:  {exog_feature_nobin_list}"
        )

        # Filter out missing columns and log a warning
        available_columns_list = [
            col for col in exog_feature_nobin_list if col in self.exog_df.columns
        ]
        missing_columns = set(exog_feature_nobin_list) - set(available_columns_list)

        if missing_columns:
            logging.info(
                f"The following columns are missing in the dataframe: {missing_columns}"
            )

        # Create a copy of dataframe containing the columns not requiring binning
        exog_no_bin_df = self.exog_df[available_columns_list].copy()

        logging.debug(
            f"Extracted dataframe: {exog_no_bin_df.shape} with columns {exog_no_bin_df.columns}"
        )

        return exog_no_bin_df

    def _create_exog_bin_dataframe(self) -> Union[None, pd.DataFrame]:
        """Creates a DataFrame for binned exogenous features.

        Retrieves a list of exogenous features from the configuration file that require binning, applies binning to each feature, and combines them into a new DataFrame.

        Args:
            None

        Raises:
            None

        Returns:
            pd.DataFrame: A DataFrame containing the binned exogenous features if exog_bin_list is not empty. Else, return None
        """
        # Retrieve the list of exogenous features to be binned
        exog_bin_list = conf_params["feature_engineering"]["exog_feature_bin"]

        if not exog_bin_list:
            logging.warning(
                "No exogenous features found in the configuration forbinning."
            )
            return None

        logging.info(f"Creating exogenous features dataframe (bin): {exog_bin_list}")

        # Initialize an empty DataFrame for binned features to be created subsequently
        exog_bin_df = pd.DataFrame()

        # Bin the specified exogenous features by using another function
        for exo_feature in exog_bin_list:
            binned_feature = self._create_binned_feature_series(exo_feature)
            if binned_feature is not None:
                new_bin_features = "_".join(["cat", str(exo_feature)])
                exog_bin_df[new_bin_features] = binned_feature

                # Update binned features information
                self.added_features_exog.extend([new_bin_features])
            else:
                continue

        logging.debug(
            f"Created dataframe: {exog_bin_df.shape} with columns {exog_bin_df.columns}"
        )

        return exog_bin_df

    def _create_binned_feature_series(self, feature_name: str) -> Optional[pd.Series]:
        """Creates binned features from the provided DataFrame.

        Args:
            feature_name (str): Name of the feature to be binned.

        Returns:
            pd.Series: A series containing the binned features, or None if invalid.
        """
        # Input Validation
        if not feature_name:
            logging.error("The feature_name parameter must not be empty.")
            return None

        if feature_name not in self.exog_df.columns:
            logging.error(
                f"The feature '{feature_name}' does not exist in the provided DataFrame."
            )
            return None

        # Retrieves and validates bin edges and labels from the configuration
        bin_edges_list, bin_labels_list = self._fetch_and_validate_config_bin_params(
            feature_name
        )
        if bin_edges_list is None or bin_labels_list is None:
            return None

        # Applies binning to the feature
        logging.info(f"Adding binned feature: cat_{feature_name}.")
        try:
            binned_feature = pd.cut(
                x=self.exog_df[feature_name],
                bins=bin_edges_list,
                labels=bin_labels_list,
                include_lowest=True,
                right=False,
            )

            logging.info(
                f"Created binned feature: cat_{feature_name} with pandas cut method."
            )

            return binned_feature

        except ValueError:
            logging.error(f"An error occurred while binning the feature: {str(e)}")
            return None

    def _fetch_and_validate_config_bin_params(
        self, feature_name: str
    ) -> Tuple[Optional[List], Optional[List]]:
        """
        Retrieves and validates bin edges and labels from the configuration.

        Args:
            feature_name (str): Name of the feature for which bin edges and labels are to be retrieved.

        Returns:
            Tuple[Optional[List], Optional[List]] : A tuple containing two lists - bin edges and bin labels, or (None, None) if invalid.
        """
        bin_edges_list = conf_params["feature_engineering"][feature_name]["bins"]
        bin_labels_list = conf_params["feature_engineering"][feature_name]["labels"]

        if not bin_edges_list or not bin_labels_list:
            logging.error("Bin edges and labels must be provided in the configuration.")
            return None, None

        if len(bin_edges_list) - 1 != len(bin_labels_list):
            logging.error(
                "Number of bin labels must be one less than the number of bin edges."
            )
            return None, None

        return bin_edges_list, bin_labels_list

    def add_is_raining_feature(
        self, rainfall_column: str, rain_threshold: float
    ) -> pd.DataFrame:
        """generate is_raining feature which is True if it is raining and False if otherwise. According to weather.gov.sg, a day is considered to have rained if the total rainfall for that day is 0.2mm or more.

        Args:
            rainfall_column (str): column name of total daily rainfall
            rain_threshold (float): set to 0.2mm as specified by weather.gov.sg

        Returns:
            pd.DataFrame: dataframe containing is_raining feature
        """
        new_feature_name = "is_raining"
        is_raining_df = self.exog_df[[rainfall_column]]
        is_raining_df[new_feature_name] = is_raining_df[rainfall_column].map(
            lambda x: True if x >= rain_threshold else False
        )
        is_raining_df.drop([rainfall_column], axis=1, inplace=True)
        # Update added features:
        self.added_features_exog.extend([new_feature_name])
        logging.debug(
            f"Generated dataframe: {is_raining_df.shape} with columns {is_raining_df.columns}"
        )
        return is_raining_df
