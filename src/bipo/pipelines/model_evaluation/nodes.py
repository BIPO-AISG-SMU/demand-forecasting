"""
This is a boilerplate pipeline 'model_evaluation'
generated using Kedro 0.18.11
"""
import mlflow
import pandas as pd
import numpy as np
import logging
from kedro_datasets.pickle import PickleDataSet
from statsmodels.miscmodels.ordinal_model import OrderedModel
from interpret.glassbox import ExplainableBoostingClassifier
from typing import Tuple, Union, Dict, Any, Callable
from bipo import settings
from sklearn.metrics import (
    precision_score,
    recall_score,
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from kedro.config import ConfigLoader

logger = logging.getLogger(settings.LOGGER_NAME)
conf_loader = ConfigLoader(conf_source=settings.CONF_SOURCE)
conf_constants = conf_loader.get("constants*")


def load_data_fold(
    partitioned_input: Dict[str, pd.DataFrame], params_dict: Dict[str, Any]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Function which loads data from a Kedro partitioned input

    Args:
        partitioned_input (Dict[str, pd.DataFrame]): Kedro IncrementalDataSet dictionary containing fold-based consolidated outlet features with suffixes _X (predictor features) and _y (predicted features).
        params_dict (Dict[str, Any]): Dictionary containing parameters referenced parameters.yml.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame] containing:
            - X_df: Dataframe containing predictor features
            - y_df: DataFrame containing predicted features
    """
    # Get parameters from dictionary. Contains filepath to reference as part of training.

    # From parameters.yml
    fold = str(params_dict["fold"])
    X_df = pd.DataFrame()
    y_df = pd.DataFrame()
    split_approach = str(params_dict["split_approach_source"])

    # Construct path using template (based on file name in data/models_specific_preprocessing)
    for partition_id, partition_df in partitioned_input.items():
        if X_df.empty or y_df.empty:
            if fold in partition_id and split_approach in partition_id:
                if partition_id.endswith("_X"):
                    X_df = partition_df
                elif partition_id.endswith("_y"):
                    y_df = partition_df
        else:
            # Terminate loop when both X and y are empty
            logger.info("Segregated fold into X(feature) and y(target).")
            break

    logger.info(f"Loaded dataframes, shapes of X: {X_df.shape}, y:{y_df.shape}")

    return X_df, y_df


def predict(
    X_df: pd.DataFrame,
    params_dict: Dict[str, Any],
    estimator: Union[OrderedModel, ExplainableBoostingClassifier],
) -> np.ndarray:
    """Function that loads a provided model and conducts prediction on a test dataset. As the model can be either orderedmodel or explanable boosting machine (ebm), necessary prediction process are governed by a conditional based on params_dict.

    Args:
        X_df (pd.DataFrame): Dataframe containing predictor features.
        params_dict (Dict[str, Any]): Dictionary referencing parameters.yml.
        estimator (Union[OrderedModel, ExplainableBoostingClassifier]): model estimator loaded from pkl file as specified in data catalog.

    Returns:
        NDArray: Numpy array containing prediction class.
    """

    # Retrieve model info and the name of target which was binned
    model_name = params_dict["model"]
    target_feature_name = params_dict["fe_target_feature_name"]

    # Get encoding labels of the target from on binning_dict which would consist the labels used for target feature which is binned.
    bin_labels_list = params_dict["binning_dict"][target_feature_name]

    # Check validity of entries. Assign constants from default
    valid_model_name_list = conf_constants["modeling"]["valid_model_name"]
    if model_name not in valid_model_name_list:
        logger.error(
            f"Invalid mode detected and does not belong to either of  {valid_model_name_list}"
        )
        logger.info(
            f"Using default model name: {conf_constants['modeling']['model_name_default']}"
        )
        model_name = conf_constants["modeling"]["model_name_default"]

    logger.info(f"Predicting test data {X_df.shape}")

    #  Ordered_model
    if model_name == "ordered_model":
        predicted_probs = estimator.model.predict(
            estimator.params, exog=X_df, which="prob"
        )
        y_pred = predicted_probs.argmax(1)

    # EBM model
    else:
        y_pred_prob = estimator.predict_proba(X_df).astype(float)

        # Take the highest probability class
        y_pred = y_pred_prob.argmax(1)
        # replace predicted y ordinal values with target categories (e.g "Low", "Medium", "High", "Very High")
        # logger.info(f"Predicted probability {y_pred}")

    target_mapping = {index: value for index, value in enumerate(bin_labels_list)}
    y_bin_labels = [*map(target_mapping.get, y_pred)]

    logger.info(f"Generated predictions from {model_name}")

    return y_bin_labels


def evaluate(
    y_pred: pd.Series, y_actual: pd.DataFrame, params_dict: Dict[str, Any]
) -> Dict[str, float]:
    """This function utilises precision, recall and accuracy metrics for evaluating model prediction against ground truth.

    Args:
        y_pred (pd.DataFrame): Dataframe containing predicted outputs.
        y_actual (pd.DataFrame): Dataframe containing test dataset.
        params_dict (Dict): Dictionary referencing parameters.yml.

    Raises:
        None

    Returns:
        Dict[str, float]: Dictionary containing evaluation matrix involving precision, recall and accuracy.
    """
    logger.info(f"Evaluating predicted {len(y_pred)} against {len(y_actual)}")
    # Instantiate dictionary with parameters
    evaluation_matrix = {}
    evaluation_matrix["precision"] = precision_score(
        y_actual, y_pred, average="weighted"
    )
    evaluation_matrix["recall"] = recall_score(y_actual, y_pred, average="weighted")
    evaluation_matrix["accuracy"] = accuracy_score(y_actual, y_pred)
    evaluation_matrix["f1"] = f1_score(y_actual, y_pred, average="weighted")

    # Log metrics to MLflow
    enable_mlflow = params_dict["enable_mlflow"]
    if enable_mlflow:
        run = mlflow.last_active_run()
        run_id = run.info.run_id
        metrics_to_log = {
            "precision": evaluation_matrix["precision"],
            "recall": evaluation_matrix["recall"],
            "accuracy": evaluation_matrix["accuracy"],
            "f1": evaluation_matrix["f1"],
        }
        with mlflow.start_run(run_id=run_id):
            mlflow.log_metrics(metrics_to_log)

    return evaluation_matrix
