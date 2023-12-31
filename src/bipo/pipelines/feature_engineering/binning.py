from typing import List, Dict
import pandas as pd
from kedro.config import ConfigLoader
from bipo import settings
import logging

logger = logging.getLogger(settings.LOGGER_NAME)

conf_loader = ConfigLoader(settings.CONF_SOURCE)


def equal_freq_binning_transform(
    df: pd.DataFrame,
    bin_column_name: str,
    bin_labels_list: List,
    bin_edges_list: List,
) -> pd.DataFrame:
    """Function that creates equal frequency binning on a specified column of interest with learned binning parameters using pandas qcut method to divide the data evenly into each bin.

    Args:
        df (pd.DataFrame): Dataframe to be processed.
        bin_column_name (str): String representing column to be binned.
        bin_labels_list (List): List of bin labels to use when binning.
        training_testing_mode (str): Mode of operating affecting normalization process.
        bin_edges_list (List): List of fitted binning parameters identified by fold and outlet.

    Raises:
        None.
    Returns:
        pd.DataFrame: Dataframe with created binned features.

    """
    try:
        new_column_name = f"binned_{bin_column_name}"
        df[new_column_name] = pd.cut(
            df[bin_column_name],
            bins=bin_edges_list,
            right=False,  # includes the rightmost edge or not
            labels=bin_labels_list,
            include_lowest=True,
            retbins=False,
            duplicates="raise",
        )
        logger.info("Binning applied with learned binned edges parameters.")

    except KeyError:
        log_string = f"Attempt to access {bin_column_name}, but was not in the  dataframe to be processed. Return as it is."
        logger.error(log_string)

    except ValueError:
        log_string = f"ValueError: {bin_column_name} is not numerical hence the requested binning could not executed. Skipping binning process."
        logger.error(log_string)
    return df


def equal_freq_binning_fit(
    df: pd.DataFrame,
    bin_column_name: str,
    bin_labels_list: List,
    fold: str,
    outlet: str,
    binning_encoding_dict: Dict[str, List],
) -> Dict[str, List]:
    """Function which extracts learned binning parameters on a specific column through pandas qcut approach which divides the data evenly.

    Args:
        df (pd.DataFrame): Dataframe to be processed.
        bin_column_name (str): String representing column to be binned.
        bin_labels_list (List): List of bin labels to use when binning.
        fold (str): input string for saving window splits artifacts. Same as partition_id in pipelines.py
        outlet (str): to assign the values to each outlets.
        training_testing_mode (str): Mode of operating affecting normalization process.
        binning_encoding_dict (Dict[str, List]): Dictionary to store learned binning parameters in a list identified by fold and outlet.

    Raises:
        None.

    Returns:
        Dict: Updated dictionary with learned binning parameters.

    """

    try:
        # Apply equal frequency binning with bin cuts information
        _, bins = pd.qcut(
            df[bin_column_name],
            q=len(bin_labels_list),
            labels=bin_labels_list,
            retbins=True,
        )

        bin_edges_list = list(bins)

        # We want the left edge of bin to be 0, since this fucntion is used for proxy revenue binning
        bin_edges_list[0] = 0
        # We want the right edge of bin to be as large as possible, since this fucntion is used for proxy revenue binning
        bin_edges_list[-1] = float("inf")
        # update the bin_edge_dict
        binning_encoding_dict[f"{fold}_{outlet}"] = bin_edges_list

        logger.info(
            f"Saved bin edges {bin_edges_list} as artefacts for {fold} and outlet: {outlet}"
        )

    except KeyError:
        log_string = f"Attempt to access {bin_column_name}, but was not in the  dataframe to be processed. Skipping processing."
        logger.error(log_string)

    except ValueError:
        log_string = f"{bin_column_name} is not numerical or the requested binning parameter extraction could not executed due to values exceeding corner bins' edge values. Skipping processing."
        logger.error(log_string)

    return binning_encoding_dict
