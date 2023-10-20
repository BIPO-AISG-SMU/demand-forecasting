"""
This is a boilerplate pipeline 'model_evaluation'
generated using Kedro 0.18.11
"""
from .nodes import load_data_fold, predict, evaluate

from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline
from kedro.config import ConfigLoader
from bipo import settings

conf_loader = ConfigLoader(conf_source=settings.CONF_SOURCE)
conf_constants = conf_loader.get("constants*")


def create_pipeline(**kwargs) -> Pipeline:
    model = conf_loader["parameters"]["model"].lower()  # Name of model
    valid_model_list = conf_constants["modeling"]["valid_model_name"]

    # The use of X_test and y_test namings are actually a generalised names for validation or test data depending on the context which this pipeline is being executed. The intent is to allow the reusability of the pipeline
    eval_train_pipeline_instance = pipeline(
        [
            node(
                func=load_data_fold,  # In nodes.py
                inputs=["model_specific_preprocessing_train", "parameters"],
                outputs=["X_train", "y_train"],  # Memorydataset
                name="model_eval_load_data_fold_train",
            ),
            node(
                func=predict,  # In nodes.py
                inputs=["X_train", "parameters", "model_training_artefact"],
                outputs="y_train_pred",
                name="model_eval_predict_train",
            ),
            node(
                func=evaluate,  # In nodes.py
                inputs=["y_train_pred", "y_train", "parameters"],
                outputs="model_evaluation_train_result",
                name="model_eval_evaluate_train",
            ),
        ],
        tags=["cleanup"],
    )

    eval_test_pipeline_instance = pipeline(
        [
            node(
                func=load_data_fold,  # In nodes.py
                inputs=["model_specific_preprocessing_validation", "parameters"],
                outputs=["X_val", "y_val"],  # Memorydataset
            ),
            node(
                func=predict,  # In nodes.py
                inputs=["X_val", "parameters", "model_training_artefact"],
                outputs="y_val_pred",
                name="model_eval_predict_val",
            ),
            node(
                func=evaluate,  # In nodes.py
                inputs=["y_val_pred", "y_val", "parameters"],
                outputs="model_evaluation_val_result",
                name="model_eval_evaluate_val",
            ),
        ],
    )

    # Define namespaced control pipelines for training and testing sections.
    eval_train_model_pipeline = pipeline(
        pipe=eval_train_pipeline_instance,
        namespace=model,  # Governs input/output to be namespaced controlled
    )

    eval_test_model_pipeline = pipeline(
        pipe=eval_test_pipeline_instance,
        namespace=model,  # Governs input/output to be namespaced controlled
    )

    return eval_train_model_pipeline + eval_test_model_pipeline
