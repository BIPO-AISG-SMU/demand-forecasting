import pandas as pd
import sys
import os
from pathlib import Path
import re
from datetime import datetime
from kedro.config import ConfigLoader
from kedro.io import DataCatalog
import tsfresh
import numpy as np
import json
from bipo.utils import get_logger, get_project_path
import logging

logging = logging.getLogger(__name__)
# Instantiate config
project_path = get_project_path()
conf_loader = ConfigLoader(conf_source=project_path / "conf")
conf_params = conf_loader["parameters"]
conf_catalog = conf_loader.get("catalog*", "catalog/**")
catalog = DataCatalog.from_config(conf_catalog)

# CONFIGS information
raw_target_feature = conf_params["general"]["target_feature"]
TARGET_FEATURE = f"binned_{raw_target_feature}"
TSFRESH_DAYS_PER_GROUP = conf_params["feature_engineering"]["endo"][
    "tsfresh_days_per_group"
]
BIN_LABELS = conf_params["feature_engineering"]["endo"]["bin_labels_list"]


class TsfreshFe:
    """Auto feature engineering with tsfresh library. Engineer features for numerical features only.
    - extract_features can either feature engineer all possible features or only relevant features which are predefined in a json file.
    - Filter only the most important engineered features and save them into a json file so that the time taken for feature engineering can be reduced, since only relevant features are engineered.

    The full list of possible engineered features can be found at https://tsfresh.readthedocs.io/en/latest/text/list_of_features.html.
    """

    def __init__(self, df: pd.DataFrame, entity: str):
        """Initialize with dataframe after endo and exo feature engineering.

        Args:
            df (pd.DataFrame): dataframe containing engineered endo and exo features.
            entity (str): Takes 2 values either 'week' or 'week_sma'.
            - If 'week', then generate features based on the number of days as specified by days_per_group, in a static fashion. All rows in each group will have the same values.
            - If 'week_sma', then generate features based on the past X days as specified by days_per_group, in a rolling fashion. All will have different values, as values are generated in a rolling window of X days. This will result in X-1 missing rows from the start of the dataframe.
        """
        self.df = df
        self.entity = entity

    def extract_features(
        self,
        extract_relevant: bool,
        days_per_group: int,
    ) -> pd.DataFrame:
        """Auto feature engineering using tsfresh library.
        If extract_relevant is True, then only extract relevant predefined features in the json file. If extract_relevant is False, then extract all possible features.

        Args:
            extract_relevant (bool): If True, extract only predefined relevant features. If False, extract all features.
            days_per_group (int): window size for feature engineering.

        Returns:
            pd.DataFrame: dataframe with features generated using tsfresh
        """

        # check if days_per_group will result in a group with only 1 row of values. If so then days_per_group is increased by 1, as a minimum of 2 rows in each group are required for extracting features.
        if len(self.df) % days_per_group == 1:
            days_per_group += 1
        # extract relevant features if extract_relevant is True
        tsfresh_parameters = None
        if extract_relevant:
            tsfresh_parameters = catalog.load("tsfresh_relevant_features")
        # select only numerical features (which drops categorical target feature) and drop columns with na values.
        numeric_df = self.df.select_dtypes(include=["number"])
        # drop rows with nan values
        self.null_rows_index_list = numeric_df[numeric_df.isnull().any(axis=1)].index
        numeric_df.drop(self.null_rows_index_list, inplace=True)
        logging.info(
            f"Removing {len(self.null_rows_index_list)} rows due to missing values as a result of 0 values in lagged proxy revenue"
        )

        if self.entity == "week":
            # generate id column
            self.generate_id_df = (
                tsfresh.utilities.dataframe_functions.add_sub_time_series_index(
                    numeric_df, sub_length=days_per_group
                )
            )
        elif self.entity == "week_sma":
            numeric_df["idx"] = 1
            # reset index and create date column from index
            numeric_df.reset_index(inplace=True)
            # generate id column by generating additional repeated rows in a rolling window fashion.
            self.generate_id_df = (
                tsfresh.utilities.dataframe_functions.roll_time_series(
                    numeric_df,
                    column_id="idx",
                    column_sort="date",
                    max_timeshift=TSFRESH_DAYS_PER_GROUP - 1,
                    min_timeshift=TSFRESH_DAYS_PER_GROUP - 1,
                )
            )
            # drop date column if not extract_features function cannot work.
            self.generate_id_df.drop(["date"], axis=1, inplace=True)
        # ensure column_id is str type
        self.generate_id_df["id"] = self.generate_id_df["id"].astype(str)

        # extract features
        extracted_features = tsfresh.extract_features(
            self.generate_id_df,
            column_id="id",
            kind_to_fc_parameters=tsfresh_parameters,
            # n_jobs=2,
        )
        return extracted_features

    def get_target_labels(self) -> pd.Series:
        """Return target labels for feature selection based on the entity.

        Args:
            None

        Returns:
            pd.Series: y target labels
        """
        if self.entity == "week":
            # get mode of target label for each group
            self.generate_id_df[TARGET_FEATURE] = self.df[TARGET_FEATURE]
            y = (
                self.generate_id_df.groupby("id")[TARGET_FEATURE]
                .agg(pd.Series.mode)
                .reset_index(drop=True)
            )
            # for cases where the mode returns multiple target labels, replace with the first target label (index 0).
            for target_label in range(len(y)):
                if not isinstance(y[target_label], str):
                    y[target_label] = y[target_label][0]
        elif self.entity == "week_sma":
            # since numeric_df dropped rows, and it is used for extracting features, self.df has to drop the equivalent rows
            self.df.drop(self.null_rows_index_list, inplace=True)
            y = (
                self.df[TARGET_FEATURE]
                .iloc[: -(TSFRESH_DAYS_PER_GROUP - 1)]
                .reset_index(drop=True)
            )
        return y

    def extract_dates_from_strings(self, string_list):
        """helper function to extract date from string. E.g given "(1, Timestamp('2021-02-07 00:00:00'))" the date '2021-02-07' is extracted.

        Args:
            string_list (list): list containing string index

        Returns:
            list: list of dates
        """
        dates = []
        date_pattern = re.compile(r"\d{4}-\d{2}-\d{2}")

        for string in string_list:
            date_match = date_pattern.search(string)
            if date_match:
                date_str = date_match.group()
                date_object = datetime.strptime(date_str, "%Y-%m-%d").date()
                dates.append(date_object)

        return dates

    def process_extracted_features(
        self, extracted_features_df: pd.DataFrame
    ) -> pd.DataFrame:
        """process extracted features based on the entity and map the engineered features to their respective datetime index.

        Args:
            extracted_features_df (pd.DataFrame): dataframe of extracted features.

        Returns:
            pd.DataFrame: processed dataframe of engineered features that have been mapped to their respective datetime indexes.
        """
        # merge extracted features to original dataframe
        if self.entity == "week":
            processed_features_df = pd.merge(
                self.generate_id_df,
                extracted_features_df,
                left_on="id",
                right_index=True,
            )
            # filter for engineered features
            processed_features_df = processed_features_df[extracted_features_df.columns]
            return processed_features_df

        elif self.entity == "week_sma":
            # convert date index to proper format
            date_index = extracted_features_df.index
            # updated_index = [
            #     index[0].split(", ")[1].strip("(')") for index in date_index
            # ]
            updated_index = self.extract_dates_from_strings(date_index)
            extracted_features_df.index = updated_index
            extracted_features_df.index.name = "date"
            return extracted_features_df

    def generate_relevance_table(
        self,
        days_per_group: int,
        n_significant: int,
    ) -> pd.DataFrame:
        """Create a feature relevance table for feature selection. This will speed up the time for feature engineering as only the most important features are engineered.
        For every feature the influence on the target is evaluated by univariate tests and the p-Value is calculated. Afterwards the Benjamini Hochberg procedure which is a multiple testing procedure decides which features to keep and which to cut off (solely based on the p-values). Benjamini Hochberg procedure decreases the false discovery rate, as sometimes small p-values (less than 5%) happen by chance, which could lead to incorrect rejection of the true null hypotheses.

        Args:
            days_per_group (int): window size for feature engineering.
            n_significant (int): number of classes for which features should be statistically significant predictors to be regarded as relevant.

        Returns:
            pd.DataFrame: relevance table
        """
        extracted_features_df = self.extract_features(False, days_per_group)
        y = self.get_target_labels()
        # drop columns with na values and duplicate columns
        extracted_features_dropna = extracted_features_df.dropna(axis=1, how="any")
        extracted_features_dropna = extracted_features_dropna.loc[
            :, ~extracted_features_dropna.T.duplicated(keep="first")
        ]
        extracted_features_dropna.reset_index(drop=True, inplace=True)
        # check that all target labels are available for feature selection
        if len(np.unique(y)) != len(BIN_LABELS):
            logging.info(
                "Not all target labels are represented. Terminate feature selection."
            )
            return
        # create relevance table
        relevance_table_df = (
            tsfresh.feature_selection.relevance.calculate_relevance_table(
                extracted_features_dropna,
                y,
                ml_task="classification",
                multiclass=True,
                n_significant=n_significant,
            )
        )
        # if no relevant features, then return None
        if relevance_table_df["relevant"].sum() == 0:
            return None

        relevance_table_df = relevance_table_df[relevance_table_df.relevant]

        # get the mean of the p-value for all target classes, and sort. Lowest mean p-value is the most important feature.
        relevance_table_df["p_value"] = relevance_table_df.filter(like="p_value").mean(
            axis=1
        )
        return relevance_table_df

    def combine_relevance_table(
        self, relevance_table_list: list, num_features: int
    ) -> pd.DataFrame:
        """get common features among all the relevance tables, aggregate the p-values by getting the lowest p-values amongst all the relevance tables.

        Args:
            relevance_table_list (list): list of relevance tables
            num_features (int): number of top relevant features to keep.

        Returns:
            pd.DataFrame: a single combined relevance table
        """

        combined_table_df = (
            tsfresh.feature_selection.relevance.combine_relevance_tables(
                relevance_table_list
            )
        )
        combined_table_df.sort_values("p_value", inplace=True)
        combined_table_df = combined_table_df.iloc[:num_features]
        return combined_table_df

    def save_relevant_features(self, relevance_table_df: pd.DataFrame):
        """Save the relevant features in json file.

        Args:
            relevance_table_df (pd.DataFrame): relevance table containing the top relevant features to be saved. Only these saved features are extracted when extracting relevant features in the extract_features method.

        Returns:
            None
        """
        features = relevance_table_df["feature"].values
        kind_to_fc_parameters = tsfresh.feature_extraction.settings.from_columns(
            features
        )
        catalog.save("tsfresh_relevant_features", kind_to_fc_parameters)
        logging.info("Completed saving of relevant features in artefacts json file")
        return None

    def run_fe_pipeline(
        self,
        extract_relevant: bool,
        days_per_group: int,
    ) -> pd.DataFrame:
        """run feature engineering pipeline to extract all possible features or predefined relevant features saved in json file by calling the extract_features and process_extracted_features methods.

        Args:
            extract_relevant (bool): If True, extract relevant features saved in json file. If False, extract all possible features.
            days_per_group (int): window size for feature engineering.

        Returns:
            pd.DataFrame: processed dataframe of engineered features that have been mapped to their respective datetime indexes.
        """
        extracted_features_df = self.extract_features(extract_relevant, days_per_group)
        return self.process_extracted_features(extracted_features_df)
