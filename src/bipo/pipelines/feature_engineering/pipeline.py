from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline

from bipo import settings
from kedro.config import ConfigLoader

# Lightweightmmm processes
from .lightweightmmm import (
    extract_mmm_params_for_folds,
    generate_mmm_features_for_outlets,
)

# General feature engineering processes
from .nodes import (
    generate_lag,
    apply_feature_encoding_transform,
    apply_feature_encoding_fit,
    apply_standard_norm_fit,
    apply_standard_norm_transform,
    apply_binning_fit,
    apply_binning_transform,
)

# tsfresh processes

from .tsfresh_node import (
    run_tsfresh_feature_selection_process,
    run_tsfresh_feature_engineering_process,
)


def create_pipeline(**kwargs) -> Pipeline:
    conf_loader = ConfigLoader(conf_source=settings.CONF_SOURCE)
    conf_params = conf_loader["parameters"]

    pipeline_instance = pipeline(
        [
            ## Feature Encodings
            node(
                func=apply_feature_encoding_fit,
                inputs=[
                    "time_agnostic_feature_engineering_training",
                    "parameters",
                ],
                outputs="feature_encoding_dict",
                name="feature_engr_apply_feature_encoding_fit",
                tags=["training"],
            ),
            node(
                func=apply_feature_encoding_transform,
                inputs=[
                    "time_agnostic_feature_engineering_training",
                    "parameters",
                    "feature_encoding_dict",
                ],
                outputs="time_dependent_training_enc",
                name="feature_engr_apply_feature_encoding_transform_train",
                tags=["training"],
            ),
            node(  # Handles validation set
                func=apply_feature_encoding_transform,
                inputs=[
                    "time_agnostic_feature_engineering_validation",
                    "parameters",
                    "feature_encoding_dict",
                ],
                outputs="time_dependent_validation_enc",
                name="feature_engr_apply_feature_encoding_transform_val",
                tags=["training"],
            ),
            ## Binning nodes
            node(
                func=apply_binning_fit,
                inputs=[
                    "time_dependent_training_enc",
                    "parameters",
                ],
                outputs="binning_encodings_dict",
                name="feature_engr_apply_binning_fit",
            ),
            node(
                func=apply_binning_transform,
                inputs=[
                    "time_dependent_training_enc",
                    "parameters",
                    "binning_encodings_dict",
                ],
                outputs="equal_freq_binning_fit_training",
                name="feature_engr_apply_binning_transform_training",
            ),
            node(  # Handles validation set
                func=apply_binning_transform,
                inputs=[
                    "time_dependent_validation_enc",
                    "parameters",
                    "binning_encodings_dict",
                ],
                outputs="equal_freq_binning_fit_validation",
                name="feature_engr_apply_binning_transform_val",
            ),
            ## Normalisation nodes.
            node(
                func=apply_standard_norm_fit,
                inputs=[
                    "equal_freq_binning_fit_training",
                    "parameters",
                ],
                outputs="std_norm_encoding_dict",
                name="feature_engr_apply_standard_norm_fit",
            ),
            node(
                func=apply_standard_norm_transform,
                inputs=[
                    "equal_freq_binning_fit_training",
                    "parameters",
                    "std_norm_encoding_dict",
                ],
                outputs="feature_engineering_training",
                name="feature_engr_apply_standard_norm_transform_training",
            ),
            node(
                func=apply_standard_norm_transform,
                inputs=[
                    "equal_freq_binning_fit_validation",
                    "parameters",
                    "std_norm_encoding_dict",
                ],
                outputs="feature_engineering_validation",
                name="feature_engr_apply_standard_norm_transform_val",
            ),
        ],
        tags=["cleanup"]
    )

    lightweight_mmm_pipeline = pipeline(
        [
            node(
                func=extract_mmm_params_for_folds,
                inputs=[
                    "feature_engineering_training",
                    "parameters",
                ],
                outputs="lightweightmmm_fitted_params",
                name="feature_engr_extract_mmm_params_for_folds",
            ),
            node(
                func=generate_mmm_features_for_outlets,
                inputs=[
                    "lightweightmmm_fitted_params",
                    "feature_engineering_training",
                    "parameters",
                ],
                outputs="lightweightmmm_features_training",
                name="feature_engr_generate_mmm_features_for_outlets_training",
            ),
            node(
                func=generate_mmm_features_for_outlets,
                inputs=[
                    "lightweightmmm_fitted_params",
                    "feature_engineering_validation",
                    "parameters",
                ],
                outputs="lightweightmmm_features_validation",
                name="feature_engr_generate_mmm_features_for_outlets_val",
            ),
        ],
        tags=["lightweightmmm", "cleanup"],
    )

    tsfresh_pipeline = pipeline(
        [
            node(
                func=run_tsfresh_feature_selection_process,
                inputs=[
                    "feature_engineering_training",
                    "parameters",
                ],
                outputs="tsfresh_fitted_params",
                name="feature_engr_tsfresh_feature_selection_process",
            ),
            node(
                func=run_tsfresh_feature_engineering_process,
                inputs=[
                    "tsfresh_fitted_params",
                    "feature_engineering_training",
                    "parameters",
                ],
                outputs="tsfresh_features_training",
                name="feature_engr_run_tsfresh_feature_engineering_process_train",
            ),
            node(
                func=run_tsfresh_feature_engineering_process,
                inputs=[
                    "tsfresh_fitted_params",
                    "feature_engineering_validation",
                    "parameters",
                ],
                outputs="tsfresh_features_validation",
                name="feature_engr_run_tsfresh_feature_engineering_process_val",
            ),
        ],
        tags=["tsfresh", "cleanup"],
    )

    # lag generation pipeline
    lag_generation_pipeline = pipeline(
        [
            node(
                func=generate_lag,
                inputs=["loaded_proxy_revenue_partitioned_data", "parameters"],
                outputs="lag_features_partitions_dict",
                name="feature_engr_generate_lag",
            ),
        ],
        tags=["lag_generation"],
    )

    # Extend pipeline based on configuration of pipeline
    if conf_params["include_lightweightMMM"]:
        pipeline_instance = pipeline_instance + lightweight_mmm_pipeline

    if conf_params["include_tsfresh"]:
        pipeline_instance = pipeline_instance + tsfresh_pipeline

    return pipeline_instance + lag_generation_pipeline
