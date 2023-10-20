import pandas as pd
import numpy as np
from numpy import ndarray
from typing import Dict, Any, List
from kedro.config import ConfigLoader
from kedro.io import DataCatalog
import logging
from collections import OrderedDict
import lightweight_mmm as lw_mmm
from lightweight_mmm import lightweight_mmm
from bipo import settings

logger = logging.getLogger(settings.LOGGER_NAME)
# Instantiate config

conf_loader = ConfigLoader(settings.CONF_SOURCE)
constants = conf_loader.get("constants*")


def validate_lags(num_lags: int) -> int:
    """Function which validates the input num_lags and attempts to typecast into integer. If negative value is obtained, a default value referencing constants.yml would be used. Same applies if non-numerical value is provided.

    Args:
        num_lags (int): _description_

    Returns:
        int: Validated num_lags with corrected value.
    """
    try:
        num_lags = int(num_lags)
        if num_lags < 0:
            num_lags = constants[default_lightweightmmm_num_lags]
            logger.error(
                f"Negative num_lags detected. Overriding with default value: {num_lag}"
            )
    except ValueError:
        num_lags = constants[default_lightweightmmm_num_lags]
        logger.error(
            f"Unable to convert input lightweightMMM lag parameters into integer. Overriding with default value: {num_lag}"
        )

    return num_lags


# 1st node
def extract_mmm_params_for_folds(
    partitioned_outlet_input: Dict[str, pd.DataFrame], params_dict: Dict[str, Any]
) -> Dict[str, Dict]:
    """Function which receives fitted parameters for each training fold using a single outlet in view of common time-index and marketing values across all outlets.

    These fitted parameters are stored into a Dictionary based on fold from training data folds, which are used to generate additional marketing features.

    Args:
        partitioned_outlet_input (Dict[str, pd.DataFrame]): Kedro IncrementalDataSet dictionary containing individual outlet related features dataframe as values with filename as identifier.
        params_dict (Dict[str, Any]): Dictionary of parameters as referenced from parameters.yml.

    Returns:
        Dict[str, Dict]: Dictionary containing fold info as keys as well as fitted LightweightMMM artefacts in a dictionary as value.
    """
    logger.info("Extracting LightweightMMM parameters for each fold...")
    # Read parameters
    target_col = params_dict["fe_target_feature_name"]
    mkt_cost_col_list = params_dict["mkt_channel_list"]

    # create empty dictionary to store output artefact name and dictionary artefacts
    mmm_fold_params_dict = {}

    # Since marketing features are assumed to be applicable for all outlets, disregarding, we just need to consider the first outlet of each fold in the partition. The file name is of the form e.g <training_fold1_expanding_window_params_....csv>

    # Get unique folds
    fold_list = [filename.split("_")[1] for filename in partitioned_outlet_input]

    # returns unique fold e.g ['fold1','fold2']
    unique_fold_list = list(set(fold_list))

    # Required additional constant to avoid ValueError: Normal distribution got invalid loc parameter in lightweightmmm which uses  half-normal distribution with mean zero and standard deviation
    small_value_adjustment = 0.0001

    for current_fold in unique_fold_list:
        for partition_id, partition_load_df in partitioned_outlet_input.items():
            # Skip processing if required columns/features are not found in at least one of the dataframe as all dataframes would contain same columns/features
            if current_fold in partition_id:
                df = partition_load_df
                is_valid_mkt_col = set(mkt_cost_col_list).issubset(set(df.columns))

                # Immediately end the process since specified columns cannot be found.
                if target_col not in df.columns or not is_valid_mkt_col:
                    logger.info(
                        "Terminating lightweightMMM fit process as no relevant columns are found."
                    )
                    return mmm_fold_params_dict

                # Get target and marketing cost columns.
                logger.info(f"Processing {current_fold}")
                logger.info(f"Checking existence of null with {partition_id}")
                null_count = df[mkt_cost_col_list].isnull().sum().sum()
                if null_count:
                    logger.info(
                        "Imputing with 0s to avoid nulls causing lightweightmmm error as a resort."
                    )
                    df[mkt_cost_col_list].fillna(0, inplace=True)
                target_df = df[target_col]

                # Value adjustment to avoid 0 value causing issues due to half-norm distribution used by lightweightmmm
                marketing_df = df[mkt_cost_col_list] + small_value_adjustment

                # Sum the values to get total cost
                total_cost_df = marketing_df.sum().to_frame(name="total_cost")

                # get fitted params in a form of dictionary. Estimate will be
                mmm_fitted_params_dict = get_fitted_parameters(
                    marketing_channel_costs=marketing_df.to_numpy(),
                    total_marketing_costs=total_cost_df.to_numpy().reshape(-1),
                    target=target_df.to_numpy(),
                    channel_columns=mkt_cost_col_list,
                )
                # Assign fold info to parametrise learned parameters
                mmm_fold_params_dict[current_fold] = mmm_fitted_params_dict
                logger.info(
                    f"Completed extraction of fitted params for {current_fold}"
                )
                # Break the loop and go to next fold
                break

            # Search till first match found
            else:
                continue

    return mmm_fold_params_dict


# 2nd node
def generate_mmm_features_for_outlets(
    extracted_mmm_params_dict: Dict[str, Dict],
    partitioned_outlet_input: Dict[str, pd.DataFrame],
    params_dict: Dict[str, Any],
) -> Dict[str, Dict]:
    """Generates lightweightmmm features for each single outlet from each fold. All outlets have the same marketing features, and hence only 1 dataframe will be generated for each fold.

    Args:
        extracted_mmm_params_input (Dict[str, Dict])): Dictionary containing fold-based LightweightMMM fitted parameters artefacts
        partitioned_outlet_input Dict[str, pd.DataFrame]): Kedro IncrementalDataSet dictionary containing individual outlet related features dataframe as values with filename as identifier.
        params_dict (Dict[str, Any]): Dictionary of parameters as referenced from parameters.yml.

    Returns:
        Dict[str, Dict]: Dictionary contain dictionaries of extracted lightweightMMM features by outlet.
    """

    # Read from parameters yml
    mmm_lags = params_dict["lightweightmmm_num_lags"]
    adstock_normalise = params_dict["lightweightmmm_adstock_normalise"]
    optimise_parameters = params_dict["lightweightmmm_optimise_parameters"]
    lightweight_params_dict = params_dict["lightweightmmm_params"]
    mkt_cost_col_list = params_dict["mkt_channel_list"]

    # Validate specified lags
    mmm_lags = validate_lags(num_lags=mmm_lags)

    lightweightmmm_features_dict = {}

    if not extracted_mmm_params_dict:
        logger.info(
            "No lightweightMMM artefacts available. Returning empty features dictionary."
        )
        return lightweightmmm_features_dict

    # iterate over fold-based outlet partitions
    for partition_id, partition_outlet_df in partitioned_outlet_input.items():
        # Get the current fold and retrieve corresponding fold based parameters
        fold = partition_id.split("_")[1]
        # get fitted parameters for specific fold
        logger.info(f"Start lightweightmmm feature engineering for {fold}")
        outlet_df = partition_outlet_df
        # if optimise_parameters is True, use optimized parameters. Otherwise, use user-defined parameters in config file
        if optimise_parameters:
            artefact_dict = extracted_mmm_params_dict[fold]
        else:
            artefact_dict = lightweight_params_dict
        # generate lightweightmmm features
        mmm_features_df = generate_mmm_features(
            lightweightmmm_params_artefact=artefact_dict,
            outlet_df=outlet_df,
            mmm_lags=mmm_lags,
            adstock_normalise=adstock_normalise,
            mkt_cost_col_list=mkt_cost_col_list,
        )
        lightweightmmm_features_dict[fold] = mmm_features_df
    logger.info(f"Completed LightweightMMM feature generation.\n")

    return lightweightmmm_features_dict


def generate_mmm_features(
    lightweightmmm_params_artefact: Dict,
    outlet_df: pd.DataFrame,
    mmm_lags: int,
    adstock_normalise: bool,
    mkt_cost_col_list: List,
) -> pd.DataFrame:
    """Generate lightweightmmm features for a single outlet as called by generate_mmm_features_for_outlets. This includes 2 features - adstock and carryover. This is used for inference (1 outlet), and used as a helper function in run_generate_mmm_features_pipeline during training (multiple outlets).

    Args:
        lightweightmmm_params_artefact (Dict): Fitted parameters artefact stored as dictionary using Kedro's  JSONDatset datatype
        outlet_df (pd.DataFrame): inference dataframe
        mmm_lags (int): Lag periods applied for mmm feature generation
        adstock_normalise (bool): True to normalise adstock values. Otherwise False.
        mkt_cost_col_list (List): list of marketing daily cost column names

    Returns:
        pd.DataFrame: dataframe containing generated lightweightmmm features
    """
    marketing_df = outlet_df[mkt_cost_col_list]
    marketing_array = marketing_df.to_numpy()
    adstock_df = lw_mmm.media_transforms.adstock(
        marketing_array,
        lag_weight=np.array(lightweightmmm_params_artefact["lag_weight"]),
        normalise=adstock_normalise,
    )
    carryover_df = lw_mmm.media_transforms.carryover(
        marketing_array,
        ad_effect_retention_rate=np.array(
            lightweightmmm_params_artefact["ad_effect_retention_rate"]
        ),
        peak_effect_delay=np.array(lightweightmmm_params_artefact["peak_effect_delay"]),
        number_lags=mmm_lags,
    )
    # convert adstock and carryover array to df, and rename cols
    adstock_df = pd.DataFrame(
        adstock_df, columns=[f"adstock_{col}" for col in mkt_cost_col_list]
    )
    adstock_df.index = marketing_df.index
    carryover_df = pd.DataFrame(
        carryover_df, columns=[f"carryover_{col}" for col in mkt_cost_col_list]
    )
    carryover_df.index = marketing_df.index

    # process adstock and carryover df
    adstock_df = process_mmm_df(marketing_df, adstock_df)
    carryover_df = process_mmm_df(marketing_df, carryover_df)

    # merge adstock_df and carryover_df and add to output dictionary
    mmm_features_df = adstock_df.join(carryover_df)
    return mmm_features_df


def process_mmm_df(marketing_df: pd.DataFrame, mmm_df: pd.DataFrame) -> pd.DataFrame:
    """process mmm dataframes which include adstock and carryover. Previous day's adstock/carryover values are added to the current day's daily cost to get the final daily cost which takes into account adstock/carryover effects.

    Args:
        marketing_df (pd.DataFrame): Marketing dataframe to be processed.
        mmm_df (pd.DataFrame): mmm dataframe, which is either adstock/carryover dataframe

    Returns:
        pd.DataFrame: processed mmm df
    """
    merged_df = marketing_df.join(mmm_df)
    # get adstock/carryover and daily cost columns
    mmm_col_list = mmm_df.columns.tolist()
    daily_cost_col_list = marketing_df.columns.tolist()
    # shift mmm columns by 1, and fillna with 0 as a result of the shift
    merged_df[mmm_col_list] = merged_df[mmm_col_list].shift(1)
    merged_df[mmm_col_list] = merged_df[mmm_col_list].fillna(0)
    # create empty dataframe to store newly generated columns
    result_df = pd.DataFrame()
    # iterate over each marketing channel
    for col1, col2 in zip(mmm_col_list, daily_cost_col_list):
        result_df[col1] = merged_df[col1] + merged_df[col2]
    return result_df


def get_fitted_parameters(
    marketing_channel_costs: ndarray,
    total_marketing_costs: ndarray,
    target: ndarray,
    channel_columns: list,
    weekday_seasonality: bool = True,
    seasonality_frequency: int = 365,
) -> Dict[str, List]:
    """Function which gets fitted parameters based on training fold marketing costs and returns 3 parameters.

    Lag_weight is extracted from fitted adstock model, while ad_effect_retention_rate and peak_effect_delay are extracted from fitted carryover model.

    Args:
        marketing_channel_costs (ndarray): individual marketing channel daily costs
        total_marketing_costs (ndarray): total cost of each marketing channel
        channel_columns (list): list of marketing channel columns
        weekday_seasonality (bool): Whether to estimate a weekday (7) parameter as per LightweightMMM definition.
        seasonality_frequency (int): Frequency of the time period used. Default is 52 as in 52 weeks per year as per LightweightMMM definition.

    Returns:
        Dict[str, List]: dictionary containing parameter name as keys, and the corresponding parameters as values.
    """
    # fit adstock model
    mmm_adstock = lightweight_mmm.LightweightMMM(model_name="adstock")
    logger.info("Fitting adstock model")
    mmm_adstock.fit(
        media=marketing_channel_costs,
        media_prior=total_marketing_costs,
        target=target,
        media_names=channel_columns,
        seed=321,
        weekday_seasonality=weekday_seasonality,
        seasonality_frequency=seasonality_frequency,
    )
    # fit carryover model
    mmm_carryover = lightweight_mmm.LightweightMMM(model_name="carryover")
    logger.info("Fitting carryover model")
    mmm_carryover.fit(
        media=marketing_channel_costs,
        media_prior=total_marketing_costs,
        target=target,
        media_names=channel_columns,
        seed=321,
        weekday_seasonality=weekday_seasonality,
        seasonality_frequency=seasonality_frequency,
    )
    # create dictionary to store parameters
    params_dict = {}
    # extract parameters from adstock and carryover models
    lag_weight = mmm_adstock._mcmc.get_samples()["lag_weight"].mean(axis=0)
    params_dict["lag_weight"] = lag_weight.tolist()
    ad_effect_retention_rate = mmm_carryover._mcmc.get_samples()[
        "ad_effect_retention_rate"
    ].mean(axis=0)
    peak_effect_delay = mmm_carryover._mcmc.get_samples()["peak_effect_delay"].mean(
        axis=0
    )
    params_dict["ad_effect_retention_rate"] = ad_effect_retention_rate.tolist()
    params_dict["peak_effect_delay"] = peak_effect_delay.tolist()
    return params_dict
