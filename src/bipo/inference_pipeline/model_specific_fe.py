# Imports
from sklearn.preprocessing import OrdinalEncoder
import pandas as pd
from datetime import date, timedelta
import logging
from typing import Union
import sys
import os
from kedro.config import ConfigLoader
from bipo.utils import get_project_path

# logging = logging.getLogger("kedro")

project_path = get_project_path()
conf_loader = ConfigLoader(conf_source=project_path / "conf")
conf_params = conf_loader["parameters"]
constants = conf_loader.get("constants*")

COLUMNS_TO_DROP = constants["inference"]["inference_columns_to_drop"]
OHE_COLUMNS_TO_DROP = constants["inference"]["ohe_columns_to_drop"]
OHE_COLUMNS_TO_ADD = constants["inference"]["ohe_columns_to_add"]
COLUMN_ORDER = constants["inference"]["column_order"]

COLUMNS_TO_RENAME = conf_params["inference"]["inference_columns_rename_map"]
ORDINAL_COLUMNS = conf_params["inference"]["ordinal_columns"]

logging = logging.getLogger(__name__)


# Layout of pipeline
class ModelSpecificFE:
    """Accepts inference data from general feature engineering output. This class will be specific for Ordered Model feature engineering. This includes dropping unnecessary columns, set dtypes, one-hot and ordinal encoding, renaming and reordering the columns."""

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def run_pipeline(self) -> pd.DataFrame:
        logging.info("Starting ordered model preprocessing")
        logging.info("Start dropping unnecessary columns")
        self.drop_columns(COLUMNS_TO_DROP)
        logging.info("Completed columns dropping")

        logging.info("Getting Numeric, categorical and ordinal columns")
        self.get_column_dtypes()
        logging.info("Starting encoding for categorical and ordinal columns")
        self.one_hot_encoding()
        self.convert_bool_to_int()
        self.apply_ordinal_encoding()
        logging.info("Completed data encoding")

        logging.info("Adding additional OHE columns")
        self.create_ohe_columns()
        logging.info("Added additional OHE columns")

        logging.info("Renaming columns for model prediction")
        self.rename_columns(COLUMNS_TO_RENAME)
        self.df = self.df.rename_axis("date")
        logging.info("Completed renaming columns")

        logging.info("Reorder columns for model prediction")
        self.df = self.df[COLUMN_ORDER]
        return self.df

    def drop_columns(self, columns_to_drop: list):
        """Drop features that are not used in training module. The features to drop are define in constants.py

        Args:
            columns_to_drop (list): features that is not needed for inference and training

        Returns:
            None
        """
        self.df.drop(columns=columns_to_drop, inplace=True)
        return None

    def get_column_dtypes(self):
        """Save all numerical, categorical and ordinal columns. For ordinal model, the order of the categories are defined.

        Args:
            None

        Returns:
            None
        """
        # add these into parameters.yml
        self.ord_columns = list(ORDINAL_COLUMNS.keys())
        # self.covid_ordered_categories = ["2", "5", "8", "10", "no limit"]

        self.int_columns = self.df.select_dtypes(include=[float, int]).columns
        # self.bool_columns = self.df.select_dtypes(include=["bool"]).columns
        self.cat_columns = self.df.select_dtypes(include=["object"]).columns
        self.ohe_columns = [x for x in self.cat_columns if x not in self.ord_columns]
        return None

    def one_hot_encoding(self):
        """One hot encode categorical columns in ohe_columns defined by get_column_dtypes and update df.
        Args:
            None
        Returns:
            None
        """
        self.df = pd.get_dummies(self.df, columns=self.ohe_columns)
        return None

    def convert_bool_to_int(self):
        """convert boolen values into int. True convert to 1 and False convert to 0.
        Args:
            None
        Returns:
            None
        """
        self.bool_columns = self.df.select_dtypes(include=["bool"]).columns
        self.df[self.bool_columns] = self.df[self.bool_columns].astype(int)
        return None

    def apply_ordinal_encoding(self):
        """Ordinal encode columns defined in ord_columns.
        Args:
            None
        Returns:
            None
        """
        for col, categories in ORDINAL_COLUMNS.items():
            ord_encoder = OrdinalEncoder(categories=[categories])
            self.df[col] = ord_encoder.fit_transform(self.df[[col]])
        return None

    def create_ohe_columns(self):
        """For ordered model, there is a need to remove a reference column after one-hot encoding. To match with features in training module, this function will add the necessary ohe columns and remove reference columns.
        Args:
            None
        Returns:
            None
        """
        for column in OHE_COLUMNS_TO_ADD:
            if column not in self.df.columns:
                self.df[column] = 0
        self.df.drop(columns=OHE_COLUMNS_TO_DROP, inplace=True)
        return None

    def rename_columns(self, rename_dict: dict):
        """Rename columns to match with columns name used to train the model.
        Args:
            rename_dict (dict): dictionary of columns to rename from constants.yml
        """
        self.df = self.df.rename(columns=rename_dict)
        return None
