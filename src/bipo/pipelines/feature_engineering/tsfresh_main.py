import pandas as pd
from datetime import datetime
import re
import tsfresh
import numpy as np
import logging
from bipo import settings

logger = logging.getLogger(settings.LOGGER_NAME)


class TsfreshFe:
    """Auto feature engineering with tsfresh library. Engineer features for numerical features only.
    - extract_features can either feature engineer all possible features or only relevant features which are predefined in a json file.
    - Filter only the most important engineered features and save them into a json file so that the time taken for feature engineering can be reduced, since only relevant features are engineered.

    The full list of possible engineered features can be found at https://tsfresh.readthedocs.io/en/latest/text/list_of_features.html.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        date_column: str,
        target_feature: str,
        days_per_group: int,
        bin_labels_list: list,
        tsfresh_features_list: list,
        shift_period: int,
        n_significant: int,
        num_features: int,
        relevant_features_dict: dict = None,
    ):
        """Initialize with dataframe after endo and exo feature engineering.

        Args:
            df (pd.DataFrame): dataframe containing engineered endo and exo features.
            date_column (str): Date feature name
            target_feature (str): target feature name
            days_per_group (int): Timeshift for *tsfresh* rolling time series' min/max timeshift parameter. Related to `tsfresh_features_list`.
            bin_labels_list (list): list of the bin labels
            tsfresh_features_list (list): list of features where tsfresh rolling time series is applied.
            shift_period (int): Number of days to shift dataframe which is equal to the earliest lagged number of days for inference (difference between last inference date and last provided lagged date)
            n_significant (int): number of predicted target classes for which features should be statistically significant predictors to be regarded as relevant.
            num_features (int): Number of derived *tsfresh* features to use based on a derived list of tsfresh's combined_relevance_tables containing list of tsfresh features for each outlet that satisfies the `tsfresh_n_significant`, based on mean aggregated p-values sorted in ascending order.
            relevant_features_dict (dict): dictionary of relevant features. Defaults to None.

        """
        self.df = df
        self.date_column = date_column
        self.target_feature = target_feature
        self.days_per_group = days_per_group
        self.bin_labels_list = bin_labels_list
        self.tsfresh_features_list = tsfresh_features_list
        self.shift_period = shift_period
        self.n_significant = n_significant
        self.num_features = num_features
        self.relevant_features_dict = relevant_features_dict

    def extract_features(self) -> pd.DataFrame:
        """Auto feature engineering using tsfresh library.

        Args:
            None

        Returns:
            pd.DataFrame: dataframe with features generated using tsfresh
        """

        # check if days_per_group will result in a group with only 1 row of values. If so then days_per_group is increased by 1, as a minimum of 2 rows in each group are required for extracting features.
        if len(self.df) % self.days_per_group == 1:
            self.days_per_group += 1
        tsfresh_parameters = self.relevant_features_dict
        # tsfresh_parameters = None
        # Drop rows based on a column subset
        self.df.dropna(subset=self.tsfresh_features_list, axis=0, inplace=True)
        logger.info(f"Datapoints after rows with null are dropped: {len(self.df)}")
        # filter for selected numerical columns
        numeric_df = self.df[self.tsfresh_features_list].copy()
        # Create a new column for tsfresh.utilities.dataframe_functions.roll_time_series to reference a column that contains no nulls
        numeric_df["idx"] = 1
        # reset index and create date column from index
        numeric_df.reset_index(inplace=True)

        # Creates sub windows of the time series. It rolls the (sorted) data frames for each kind and each id separately in the “time” domain. Refer https://tsfresh.readthedocs.io/en/latest/api/tsfresh.utilities.html. For each rolling step, a new id is created by the scheme ({id}, {shift}), here id is the former id of the column and shift is the amount of “time” shift. Note that this creates NEW ID.
        self.generate_id_df = tsfresh.utilities.dataframe_functions.roll_time_series(
            numeric_df,
            column_id="idx",
            column_sort=self.date_column,
            max_timeshift=self.days_per_group - 1,
            min_timeshift=self.days_per_group - 1,
        )

        # drop date column if not extract_features function cannot work.
        self.generate_id_df.drop(columns=self.date_column, inplace=True)
        # ensure column_id is str type
        self.generate_id_df["id"] = self.generate_id_df["id"].astype(str)
        # extract features
        extracted_features_df = tsfresh.extract_features(
            self.generate_id_df,
            column_id="id",
            kind_to_fc_parameters=tsfresh_parameters,
            # n_jobs=2,
        )
        # shift dataframe by shift_period to match inference period
        extracted_features_df = extracted_features_df.shift(periods=self.shift_period)
        return extracted_features_df

    def process_extracted_tsfresh_features_index(
        self, df: pd.DataFrame
    ) -> pd.DataFrame:
        """Function which reformats index of output extracted_features_df due to new id creation as a result of tsfresh feature generation.

        Example, (1, ' 2021-02-07') index which is returned during tsfresh feature generation is to be converted into 2021-02-07 datetime type instead.

        Args:
            df (pd.DataFrame): Dataframe containing extracted features.

        Returns:
            pd.DataFrame: processed dataframe of engineered features that have been mapped to their respective datetime indexes.
        """
        df.index = df.index.to_series().map(
            lambda x: re.sub(r"[ ()\"\']", "", x).split(",")[-1].strip()
        )

        df.index = pd.to_datetime(df.index, format="%Y-%m-%d")

        # Set index name to provided date_column name.
        df.index.name = self.date_column
        return df

    def generate_relevance_table(
        self,
    ) -> pd.DataFrame:
        """Create a feature relevance table for feature selection. This will speed up the time for feature engineering as only the most important features are engineered.
        For every feature the influence on the target is evaluated by univariate tests and the p-Value is calculated. Afterwards the Benjamini Hochberg procedure which is a multiple testing procedure decides which features to keep and which to cut off (solely based on the p-values). Benjamini Hochberg procedure decreases the false discovery rate, as sometimes small p-values (less than 5%) happen by chance, which could lead to incorrect rejection of the true null hypotheses.

        Args:
           None

        Returns:
            pd.DataFrame: relevance table
        """
        extracted_features_df = self.extract_features()
        # Reformat index of output extracted_features_df due to new id creation which returns (1, ' 2021-02-07') a string as example). This is done by removing non-alphanumeric strings (except for - & ,) before splitting and subsequently extract last entry separated by ,
        extracted_features_df = self.process_extracted_tsfresh_features_index(
            df=extracted_features_df
        )

        extracted_features_df.index = pd.to_datetime(
            extracted_features_df.index, format="%Y-%m-%d"
        )

        # Drop rows/columns with all nulls
        extracted_features_df.dropna(axis=1, how="all", inplace=True)

        extracted_features_df.dropna(axis=0, how="all", inplace=True)

        # drop columns with duplicate columns
        extracted_features_drop_duplicates = extracted_features_df.loc[
            :, ~extracted_features_df.T.duplicated(keep="first")
        ]
        # drop columns with any nulls as relevance table generation does not accept features with nan.
        extracted_features_drop_duplicates.dropna(axis=1, how="any", inplace=True)

        # Extract target feature column
        y = self.df[self.target_feature]
        # convert index to datetime to match index of extracted_features_dropna
        y.index = pd.to_datetime(y.index, format="%Y-%m-%d")
        y = y[y.index.isin(extracted_features_drop_duplicates.index)]

        # extracted_features_dropna.reset_index(drop=True, inplace=True)
        # check that all target labels are available for feature selection
        symm_diff = set(y.values).symmetric_difference(set(self.bin_labels_list))
        if symm_diff:
            logger.info(
                f"Target feature labels from data and specified binned_labels_list are not equivalent. Symmetric difference: {symm_diff}"
            )
            return None

        # Apply string to value mapping for y values to ensure correct processing of calculate_relevance_table
        # y = y.map(lambda y: self.bin_labels_list.index(y))
        relevance_table_df = (
            tsfresh.feature_selection.relevance.calculate_relevance_table(
                extracted_features_drop_duplicates,
                y,
                ml_task="classification",
                multiclass=True,
                n_significant=self.n_significant,
            )
        )
        # The dataframe relevant column indicate bool true/false on derived tssfresh features. If all are false the sum which is 0 means none of the derived features should be considered.
        if relevance_table_df["relevant"].sum() == 0:
            logger.info("No relevant feature(s) found.")
            return None

        # Extract feature which are relevant as identified by 'relevant' column
        relevance_feat_table_df = relevance_table_df[
            relevance_table_df["relevant"] == True
        ]

        # get the mean of all p_value prefixed column names and average them out across all target classes. No columns dropped as this is used by  tsfresh.feature_selection.relevance.combine_relevance_tables in tsfresh_node.py which is the caller of this function as well.
        p_value_col_list = [
            col for col in relevance_feat_table_df.columns if col.startswith("p_value_")
        ]
        relevance_feat_table_df["p_value"] = relevance_feat_table_df[
            p_value_col_list
        ].mean(axis=1)

        return relevance_feat_table_df

    def combine_relevance_table(self, relevance_table_list: list) -> pd.DataFrame:
        """get common features among all the relevance tables, aggregate the p-values by getting the lowest p-values amongst all the relevance tables. Assumes the relevance_table_list has 1 single entry

        Args:
            relevance_table_list (list): list of relevance tables

        Returns:
            pd.DataFrame: a single combined relevance table
        """
        if len(relevance_table_list) == 1:
            combined_table_df = relevance_table_list[0]
        # Create a combined relevance table out of a list of relevance tables, aggregating the p-values and the relevances.
        elif len(relevance_table_list) > 1:
            combined_table_df = (
                tsfresh.feature_selection.relevance.combine_relevance_tables(
                    relevance_table_list
                )
            )
        else:
            logger.error("List of relevance tables provided is empty")
        combined_table_df.sort_values("p_value", inplace=True)
        combined_table_df = combined_table_df.iloc[: self.num_features]
        logger.info(f"Number of relevant features: {len(combined_table_df)}")
        return combined_table_df
