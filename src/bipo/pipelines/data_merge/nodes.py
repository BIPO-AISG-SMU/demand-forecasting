"""
This is a boilerplate pipeline 'data_merge'
generated using Kedro 0.18.11
"""
import pandas as pd
import numpy as np
from typing import Dict
from datetime import datetime
from kedro.config import ConfigLoader
from bipo import settings
import os
import logging

logger = logging.getLogger(settings.LOGGER_NAME)
conf_loader = ConfigLoader(conf_source=settings.CONF_SOURCE)
const_dict = conf_loader.get("constants*")

# DEFAULT CONSTANTS
DEFAULT_DATE_COL = const_dict["default_date_col"]


def concat_outlet_preprocessed_data(
    partitioned_input: Dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """Function that loads and concatenates all outlet datafiles in partitioned_input into one single dataframe.

    Args:
        partitioned_input (Dict[str, pd.DataFrame]): Kedro IncrementalDataSet dictionary containing individual outlet related features dataframe as values with filename as identifier.

    Raises:
        None

    Returns:
        pd.DataFrame: Dataframe representing outlet.
    """
    # Set empty dataframe to prepare for concatenation
    merged_outlet_df = pd.DataFrame()
    for partition_id, partition_load_df in sorted(partitioned_input.items()):
        if "_processed" in partition_id:
            partition_data = partition_load_df
            merged_outlet_df = pd.concat(
                [merged_outlet_df, partition_data], join="outer"
            )
            logger.info(f"Concatenated {partition_id} to existing dataframe.")
        else:
            logger.info(
                f"{partition_id} file not included in the concatenation process."
            )

    # Assume date is confirmed inside
    if DEFAULT_DATE_COL in merged_outlet_df.columns:
        merged_outlet_df[DEFAULT_DATE_COL] = pd.to_datetime(
            merged_outlet_df[DEFAULT_DATE_COL]
        )
        merged_outlet_df.sort_values(
            by=[DEFAULT_DATE_COL], inplace=True
        )  # removed "cost_centre_code"

        merged_outlet_df.set_index(DEFAULT_DATE_COL, inplace=True)
    else:
        merged_outlet_df.sort_index(inplace=True)
    logger.info(f"Merged data is of shape: {merged_outlet_df.shape}\n")
    return merged_outlet_df
