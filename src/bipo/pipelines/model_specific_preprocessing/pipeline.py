from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline

# Import relevant function from nodes
from .nodes import (
    remove_unnecessary_columns_and_rows,
    identify_const_column,
    remove_const_column,
    reorder_data,
    split_data_into_X_y_features,
)

from kedro.config import ConfigLoader
from bipo import settings

conf_loader = ConfigLoader(conf_source=settings.CONF_SOURCE)
conf_constants = conf_loader.get("constants*")


def create_pipeline(**kwargs) -> Pipeline:
    model = str(conf_loader["parameters"]["model"])
    valid_model_list = conf_constants["modeling"]["valid_model_name_list"]

    # Override invalid model name if not valid
    if model not in valid_model_list:
        model = conf_constants["modeling"]["default_model"]

    model_spec_preprocessing_pipeline_instance = pipeline(
        [
            # Unnecessary column removal
            node(  # Handles training data
                func=remove_unnecessary_columns_and_rows,
                inputs=["merged_features_training", "parameters"],
                outputs="removed_columns_rows_training",
                name="model_spec_preprocess_remove_unnecessary_columns_training",
            ),
            node(  # Handles validation data
                func=remove_unnecessary_columns_and_rows,
                inputs=["merged_features_validation", "parameters"],
                outputs="removed_columns_rows_validation",
                name="model_spec_preprocess_remove_unnecessary_columns_validation",
            ),
            node(  # Handles testing data
                func=remove_unnecessary_columns_and_rows,
                inputs=["merged_features_testing", "parameters"],
                outputs="removed_columns_rows_testing",
                name="model_spec_preprocess_remove_unnecessary_columns_testing",
            ),
            # Constant column identification and removal
            node(
                func=identify_const_column,
                inputs=[
                    "removed_columns_rows_training",
                ],
                outputs="constant_column_params",
                name="model_spec_preprocess_identify_const_column",
            ),
            # Remove constants column
            node(
                func=remove_const_column,
                inputs=[
                    "removed_columns_rows_training",
                    "constant_column_params",
                ],
                outputs="remove_const_colum_data_training",
                name="model_spec_preprocess_remove_const_column_training",
            ),
            node(
                func=remove_const_column,
                inputs=[
                    "removed_columns_rows_validation",
                    "constant_column_params",
                ],
                outputs="remove_const_colum_data_validation",
                name="model_spec_preprocess_remove_const_column_validation",
            ),
            node(
                func=remove_const_column,
                inputs=[
                    "removed_columns_rows_testing",
                    "constant_column_params",
                ],
                outputs="remove_const_colum_data_testing",
                name="model_spec_preprocess_remove_const_column_testing",
            ),
            node(
                func=reorder_data,  # Combine train/val
                inputs=[
                    "remove_const_colum_data_training",
                    "remove_const_colum_data_validation",
                    "remove_const_colum_data_testing",
                ],
                outputs="reordered_data_folds",
                name="model_spec_preprocess_reorder_data",
            ),
            node(
                func=split_data_into_X_y_features,
                inputs=["reordered_data_folds", "parameters"],
                outputs=[
                    f"{model}.model_specific_preprocessing_training",
                    f"{model}.model_specific_preprocessing_validation",
                    f"{model}.model_specific_preprocessing_testing",
                ],
                name="model_spec_preprocess_split_data_into_X_y_features",
            ),
        ],
    )

    return model_spec_preprocessing_pipeline_instance
