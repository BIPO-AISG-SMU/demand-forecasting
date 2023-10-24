"""
This is a boilerplate pipeline 'model_training'
generated using Kedro 0.18.11
"""
import mlflow
import os
import requests
from datetime import datetime
import pandas as pd
import numpy as np
import logging
from statsmodels.miscmodels.ordinal_model import OrderedModel
from interpret.glassbox import ExplainableBoostingClassifier
from typing import Union, Dict, Any
from kedro.config import ConfigLoader
from bipo import settings

conf_loader = ConfigLoader(conf_source=settings.CONF_SOURCE)
conf_constants = conf_loader.get("constants*")
conf_params = conf_loader["parameters"]
logger = logging.getLogger(settings.LOGGER_NAME)


def train_model(
    partitioned_input: Dict[str, pd.DataFrame],
    params_dict: Dict[str, Any],
    model_namespace_params_dict: Dict[str, Any],
) -> Union[ExplainableBoostingClassifier, OrderedModel]:
    """Function that trains a model on a specified data fold defined by parameters.yml from a Kedro IncrementalDataSet contaning various data features identified by suffix '_X' representing predictor features and suffix '_y' representing predicted feature.

    Args:
        partitioned_input (Dict[str, pd.DataFrame]): A dictionary with partition ids as keys and dataframe as values.
        params_dict (Dict[str, Any]): Dictionary referencing parameters.yml.
        model_namespace_params_dict (Dict[str, Any]): Dictionary referencing values from parameters/model_training.yml

    Raises:
        None.

    Returns:
        Union[ExplainableBoostingClassifier, OrderedModel]: Either ExplainableBoostingClassifier or OrderedModel: Classes representing fitted model.
    """

    # Get parameters from dictionary. Contains filepath to reference as part
    model_params_dict = model_namespace_params_dict["params"]
    model_name = str(model_params_dict["model_name"])

    # From parameters.yml
    fold = str(params_dict["fold"])
    split_approach = str(params_dict["split_approach_source"])

    # Get encoding labels of the target from on binning_dict which would consist the labels used for target feature which is binned.This is based on non-binned feature
    target_feature_name = params_dict["fe_target_feature_name"]
    bin_labels_list = params_dict["binning_dict"][target_feature_name]

    target_column_for_modeling = params_dict["target_column_for_modeling"]

    # Extract config for mlflow
    enable_mlflow = params_dict["enable_mlflow"]

    X_train_df = pd.DataFrame()
    y_train_df = pd.DataFrame()

    # Convert string to integer using enumerate
    binned_target_mapping = {
        value: index for index, value in enumerate(bin_labels_list)
    }
    logger.info(f"Defined binning mapping: {binned_target_mapping}")
    # Construct path using template (based on file name in data/models_specific_preprocessing)
    for partition_id, partition_df in partitioned_input.items():
        if X_train_df.empty or y_train_df.empty:
            if fold in partition_id and split_approach in partition_id:
                if partition_id.endswith("_X"):
                    X_train_df = partition_df.copy()
                elif partition_id.endswith("_y"):
                    y_train_df = partition_df.copy()

                    # Convert y values to numeric
                    y_train_df = y_train_df[target_column_for_modeling].map(
                        binned_target_mapping, na_action=None
                    )
        else:
            # Terminate loop when both X and y are not empty
            break

    # Load X,y data
    logger.info(
        f"Loaded dataframes for training based on fold: {fold} and split approach {split_approach}, shapes of X: {X_train_df.shape}, y: {y_train_df.shape}"
    )

    # Check validity of model name
    valid_model_name_list = conf_constants["modeling"]["valid_model_name"]
    if model_name not in valid_model_name_list:
        logger.error(
            f"Invalid mode detected and does not belong to either of  {valid_model_name_list}"
        )
        logger.info(
            f"Using default model name: {conf_constants['modeling']['model_name_default']}"
        )
        model_name = conf_constants["modeling"]["model_name_default"]

    # Read parameters for instantiating necessary models
    logger.info(f"Instantiating {model_name} with necessary parameters")
    logger.info("Attempting to retrieve parameters required...")

    if model_name == "ebm":
        try:
            outer_bags = model_params_dict["outer_bags"]
            inner_bags = model_params_dict["inner_bags"]
            learning_rate = model_params_dict["learning_rate"]
            interactions = model_params_dict["interactions"]
            max_bins = model_params_dict["max_bins"]
            max_leaves = model_params_dict["max_leaves"]
            min_samples_leaf = model_params_dict["min_samples_leaf"]
            logger.info(f"Extracted parameters for ebm:{model_params_dict}")

        except KeyError:
            logger.info(
                "Unable to reference required parameters for ebm model due to key error. Using default settings..."
            )
            ebm_params_const_dict = conf_constants["modeling"]["ebm"]
            # Read off defaults from constants.yml
            outer_bags = ebm_params_const_dict["outer_bags"]
            inner_bags = ebm_params_const_dict["inner_bags"]
            learning_rate = ebm_params_const_dict["learning_rate"]
            interactions = ebm_params_const_dict["interactions"]
            max_bins = ebm_params_const_dict["max_bins"]
            max_leaves = ebm_params_const_dict["max_leaves"]
            min_samples_leaf = ebm_params_const_dict["min_samples_leaf"]
            logger.info(f"Using default parameters for ebm:{ebm_params_const_dict}")

        logger.info("Instantiating EBM")
        model = ExplainableBoostingClassifier(
            feature_names=None,
            feature_types=None,
            max_bins=max_bins,
            outer_bags=outer_bags,
            inner_bags=inner_bags,
            learning_rate=learning_rate,
            min_samples_leaf=min_samples_leaf,
            interactions=interactions,
            random_state=42,
        )

        logger.info("Applying normal fit")
        trained_model = model.fit(X_train_df, y_train_df)

        # Run MLflow tracking if enabled
        if enable_mlflow:
            mlflow_tracking(params_dict, model_params_dict)

        logger.info("Completed model training using Explanable Boosting Machine.\n")

    # For ordered model
    else:
        # Extract information from params dict
        try:
            distr = model_params_dict["distr"]
            method = model_params_dict["method"]
            max_iter = model_params_dict["max_iter"]
            logger.info(f"Extracted parameters for OrderedModel:{model_params_dict}")

        except KeyError:
            ordered_model_params_const_dict = conf_constants["modeling"][
                "ordered_model"
            ]
            logger.info(
                "Unable to reference required parameters for OrderedModel due to key error. Using default settings..."
            )
            distr = ordered_model_params_const_dict["distr"]
            method = ordered_model_params_const_dict["method"]
            max_iter = ordered_model_params_const_dict["max_iter"]
            logger.info(
                f"Using default parameters for ebm:{ordered_model_params_const_dict}"
            )

        # Instantiate and fit OrderedModel with parameters
        model = OrderedModel(y_train_df, X_train_df, distr=distr)
        logger.info(f"Instantiated OrderedModel with parameters {distr}")
        trained_model = model.fit(method=method, maxiter=max_iter, disp=True)

        # Run MLflow tracking if enabled
        if enable_mlflow:
            mlflow_tracking(params_dict, model_params_dict)
        logger.info("Completed model training using OrderedModel.\n")

    return trained_model


def explain_ebm(
    model: ExplainableBoostingClassifier,
    partitioned_input: Dict[str, pd.DataFrame] = None,
):
    """provides explainability of the ebm model and saves the plotly figure/static image locally. Can either output a png or html file which is configurable by the output_type parameter, where if it is png, a static png file is the output, and if it is html, a html file will be returned. This function is used in the training pipeline.

    1) global: feature importance for the training dataset. Term importances are the mean absolute contribution (score) each term (feature or interaction) makes to predictions averaged across the training dataset. Contributions are weighted by the number of samples in each bin, and by the sample weights (if any)
    2) feature: contribution score to predictions made by the model of the specified feature

    Args:
        model (ExplainableBoostingClassifier): ebm model weights
        partitioned_input (Dict[str, pd.DataFrame]):  Kedro IncrementalDataSet Dictionary containing fold-based training dataset containing features and target variables identified with suffixes _X and _y.

    Returns:
        plotly.graph_objs: plotly figure. Also saves the plotly figure/static image locally
    """
    # load config
    output_type = conf_params["output_type"]
    feature_list = conf_params["feature_to_explain_list"]
    output_filepath = conf_constants["modeling"]["explainability_filepath"]
    fold = str(conf_params["fold"])
    split_approach = str(conf_params["split_approach_source"])

    # make directory
    if not os.path.exists(output_filepath):
        os.makedirs(output_filepath)
    # global feature importance
    global_ebm_explanation = model.explain_global()
    global_plotly_fig = global_ebm_explanation.visualize()
    if output_type == "png":
        global_plotly_fig.write_image(
            f"{output_filepath}/global_feature_importance.png"
        )
    elif output_type == "html":
        global_plotly_fig.write_html(
            f"{output_filepath}/global_feature_importance.html"
        )
    else:
        logger.error("Invalid value for output_type. Accepts either png or html")

    # detailed individual feature importance
    for partition_id, partition_df in partitioned_input.items():
        if fold in partition_id and split_approach in partition_id:
            if partition_id.endswith("_X"):
                X_train_df = partition_df
                break
    for feature in feature_list:
        # get column index of feature
        feature_index = X_train_df.columns.get_loc(feature)
        ebm_explanation = model.explain_global()
        plotly_fig = ebm_explanation.visualize(feature_index)
        if output_type == "png":
            plotly_fig.write_image(f"{output_filepath}/{feature}_importance.png")
        elif output_type == "html":
            plotly_fig.write_html(f"{output_filepath}/{feature}_importance.html")
        else:
            logger.error("Invalid value for output_type. Accepts either png or html")
    return True


def mlflow_tracking(
    params_dict: Dict[str, Any], model_params_dict: Dict[str, Any]
) -> bool:
    """
    Track model training using MLflow and log relevant parameters.

    Args:
        params_dict (Dict[str, Any]): Dictionary referencing values from parameters.yml
        model_params_dict (Dict[str, Any]): Dictionary referencing values from parameters/model_training.yml

    Returns:
        bool: True for recognized model; False for unrecognized model name.
    """
    logger.info(f"Enabled MLflow tracking")
    # Generate a unique run_name for MLflow tracking
    model_name = model_params_dict["model_name"]
    run_timestamp = datetime.now().strftime("%m%d_%H%M%S")
    run_name = f"{model_name}-{run_timestamp}"

    # Load MLflow setting from the parameters.yml
    is_remote_mlflow = params_dict["is_remote_mlflow"]
    tracking_uri = params_dict["tracking_uri"]
    experiment_name = params_dict["experiment_name_prefix"]
    experiment_name = f"{experiment_name}-{model_name}"

    # If needed, setup environment for the remote tracking server
    if is_remote_mlflow:
        setup_remote_mlflow(tracking_uri, experiment_name)

    # Start MLflow tracking
    with mlflow.start_run():
        mlflow.set_tag("mlflow.runName", run_name)
        if model_name == "ordered_model":
            # Log specific parameters for ordered_model
            params_to_log = {
                "distr": model_params_dict["distr"],
                "method": model_params_dict["method"],
                "maxiter": model_params_dict["max_iter"],
            }
            mlflow.log_params(params_to_log)
            return True
        elif model_name == "ebm":
            # Log specific parameters for ebm
            params_to_log = {
                "outer_bags": model_params_dict["outer_bags"],
                "inner_bags": model_params_dict["inner_bags"],
                "learning_rate": model_params_dict["learning_rate"],
                "interactions": model_params_dict["interactions"],
                "max_bins": model_params_dict["max_bins"],
                "max_leaves": model_params_dict["max_leaves"],
                "min_samples_leaf": model_params_dict["min_samples_leaf"],
            }
            mlflow.log_params(params_to_log)
            return True
        else:
            logger.error(f"Unrecognized model_name: {model_name}")
            return False


def setup_remote_mlflow(remote_uri: str, experiment_name: str) -> None:
    """Function which set up remote MLflow tracking URI and experiment.

    Args:
        remote_uri (str): The desired remote MLflow tracking URI.
        experiment_name (str): The base name for the experiment.

    Raises:
        None.

    Returns:
        bool: False if the URI is unavailable for any reason
    """
    logger.info(f"Enabled remote MLflow tracking server")
    try:
        response = requests.get(remote_uri, timeout=5)
        is_uri_available = True if response.status_code == 200 else False
    except requests.RequestException:
        is_uri_available = False

    if is_uri_available:
        mlflow.set_tracking_uri(remote_uri)
        logger.info(f"Current MLflow tracking uri: {remote_uri}")

        if mlflow.get_experiment_by_name(experiment_name) is not None:
            mlflow.set_experiment(experiment_name)
            logger.info(f"Current experiment_name: {experiment_name}")
        else:
            logger.warning(
                f"Given experiment name '{experiment_name}' doesn't exist. MLflow will use the 'default' experiment name."
            )
        return True
    else:
        logger.error(
            f"Given MLflow tracking URI {remote_uri} is not reachable. The code will continue to use the 'localhost' setting."
        )
        return False
