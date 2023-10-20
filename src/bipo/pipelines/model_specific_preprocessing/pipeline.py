"""
This is a boilerplate pipeline 'model_specific_preprocessing'
generated using Kedro 0.18.11
"""
from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline

# Import relevant function from nodes
from .nodes import (
    remove_unnecessary_columns_and_rows,
    merge_fold_and_generated_lag,
    identify_const_column,
    remove_const_column,
    reorder_data,
    split_data_into_X_y_features,
    concat_same_folds_of_outlet_data,
    merge_tsfresh_features_with_outlets,
    merge_mmm_features_with_fold_outlets,
)

from kedro.config import ConfigLoader
from kedro.framework.session import KedroSession
from bipo import settings

conf_loader = ConfigLoader(conf_source=settings.CONF_SOURCE)
conf_constants = conf_loader.get("constants*")


def create_pipeline(**kwargs) -> Pipeline:
    model = str(conf_loader["parameters"]["model"])
    valid_model_list = conf_constants["modeling"]["valid_model_name"]

    # Override invalid model name if not valid
    if model not in valid_model_list:
        model = conf_constants["modeling"]["model_name_default"]

    # Pipeline facilitating merging of lightweightMMM/tsfresh feature merging with outlet features.
    merge_generated_features_pipeline = pipeline(
        [
            # Merge mmm features and fold outlets
            node(
                func=merge_mmm_features_with_fold_outlets,
                inputs=[
                    "feature_engineering_training",
                    "lightweightmmm_features_training",
                ],
                outputs="merged_lightweightmmm_training",
                name="model_spec_merge_mmm_features_with_fold_outlets_training",
                tags=["lightweightmmm"],
            ),
            node(
                func=merge_mmm_features_with_fold_outlets,
                inputs=[
                    "feature_engineering_validation",
                    "lightweightmmm_features_validation",
                ],
                outputs="merged_lightweightmmm_validation",
                name="model_spec_merge_mmm_features_with_fold_outlets_validation",
                tags=["lightweightmmm"],
            ),
            node(
                func=merge_tsfresh_features_with_outlets,
                inputs=[
                    "merged_lightweightmmm_training",
                    "tsfresh_features_training",  # From feature_engineering
                ],
                outputs="merged_tsfresh_features_training",
                name="model_spec_merge_tsfresh_features_with_fold_outlets_training",
                tags=["tsfresh"],
            ),
            node(
                func=merge_tsfresh_features_with_outlets,
                inputs=[
                    "merged_lightweightmmm_validation",
                    "tsfresh_features_validation",  # From feature_engineering
                ],
                outputs="merged_tsfresh_features_validation",
                name="model_spec_merge_tsfresh_features_with_fold_outlets_validation",
                tags=["tsfresh"],
            ),
            # Lag features merging
            node(
                func=merge_fold_and_generated_lag,
                inputs=[
                    "merged_tsfresh_features_training",
                    "lag_features_partitions_dict",  # From feature_engineering
                ],
                outputs="merged_features_training",
                name="model_spec_merge_fold_and_generated_lag_training",
                tags=["lag_generation"],
            ),
            node(
                func=merge_fold_and_generated_lag,
                inputs=[
                    "merged_tsfresh_features_validation",
                    "lag_features_partitions_dict",  # From feature_engineering
                ],
                outputs="merged_features_validation",
                name="model_spec_merge_fold_and_generated_lag_validation",
                tags=["lag_generation"],
            ),
        ],
        tags=["cleanup"],
    )

    # Pipeline which combines all outlets
    model_spec_preprocessing_pipeline_instance = pipeline(
        [
            node(
                func=concat_same_folds_of_outlet_data,
                inputs="merged_features_training",
                outputs="concatenated_folds_training",
                name="model_spec_preprocess_concat_same_folds_of_outlet_data_train",
            ),
            node(
                func=concat_same_folds_of_outlet_data,
                inputs="merged_features_validation",
                outputs="concatenated_folds_validation",
                name="model_spec_preprocess_concat_same_folds_of_outlet_data_val",
            ),
            # Unnecessary column removal
            node(
                func=remove_unnecessary_columns_and_rows,
                inputs=["concatenated_folds_training", "parameters"],
                outputs="removed_columns_training",
                name="model_spec_preprocess_remove_unnecessary_columns_train",
            ),
            node(
                func=remove_unnecessary_columns_and_rows,
                inputs=["concatenated_folds_validation", "parameters"],
                outputs="removed_columns_validation",
                name="model_spec_preprocess_remove_unnecessary_columns_val",
            ),
            # Constant column identification and removal
            node(
                func=identify_const_column,
                inputs=[
                    "removed_columns_training",
                    "parameters",
                ],
                outputs="constant_column_params",
                name="model_spec_preprocess_identify_const_column",
            ),
            # Remove constants column
            node(
                func=remove_const_column,
                inputs=[
                    "removed_columns_training",
                    "parameters",
                    "constant_column_params",
                ],
                outputs="remove_const_colum_data_training",
                name="model_spec_preprocess_remove_const_column_train",
            ),
            node(
                func=remove_const_column,
                inputs=[
                    "removed_columns_validation",
                    "parameters",
                    "constant_column_params",
                ],
                outputs="remove_const_colum_data_validation",
                name="model_spec_preprocess_remove_const_column_val",
            ),
            node(
                func=reorder_data,  # Combine train/val
                inputs=[
                    "remove_const_colum_data_training",
                    "remove_const_colum_data_validation",
                ],
                outputs="reordered_data_folds",
                name="model_spec_preprocess_reorder_data",
            ),
            node(
                func=split_data_into_X_y_features,
                inputs=["reordered_data_folds", "parameters"],
                outputs=[
                    f"{model}.model_specific_preprocessing_train",
                    f"{model}.model_specific_preprocessing_validation",
                    f"{model}.model_specific_preprocessing_test",
                ],
                name="model_spec_preprocess_split_data_into_X_y_features",
            ),
        ],
    )

    return (
        merge_generated_features_pipeline + model_spec_preprocessing_pipeline_instance
    )
