from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline

from bipo import settings
from kedro.config import ConfigLoader

# Lag feature processes
from .lag_feature_generation import merge_fold_and_generated_lag

# Lightweightmmm processes
from .lightweightmmm import (
    extract_mmm_params_for_folds,
    generate_mmm_features_for_outlets,
    merge_mmm_features_with_fold_outlets,
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
    concat_same_folds_of_outlet_data,
)

# tsfresh processes

from .tsfresh_node import (
    run_tsfresh_feature_selection_process,
    run_tsfresh_feature_engineering_process,
    merge_tsfresh_features_with_outlets,
)


def create_pipeline(**kwargs) -> Pipeline:
    conf_loader = ConfigLoader(conf_source=settings.CONF_SOURCE)
    conf_params = conf_loader["parameters"]

    feature_engineering_pipeline_instance = pipeline(
        [
            ## Binning nodes
            node(
                func=apply_binning_fit,
                inputs=[
                    "time_agnostic_feature_engineering_training",
                    "parameters",
                ],
                outputs="binning_encodings_dict",
                name="feature_engr_apply_binning_fit",
            ),
            node(
                func=apply_binning_transform,
                inputs=[
                    "time_agnostic_feature_engineering_training",
                    "parameters",
                    "binning_encodings_dict",
                ],
                outputs="equal_freq_binning_fit_training",
                name="feature_engr_apply_binning_transform_training",
            ),
            node(  # Handles validation set
                func=apply_binning_transform,
                inputs=[
                    "time_agnostic_feature_engineering_validation",
                    "parameters",
                    "binning_encodings_dict",
                ],
                outputs="equal_freq_binning_fit_validation",
                name="feature_engr_apply_binning_transform_val",
            ),
            ## Standardisation/Normalisation nodes.
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
        tags=["cleanup"],
    )

    # Pipeline for generating lightweightmmmm features only. Input continues off the last node of feature_engineering_pipeline_instance
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
    )

    # Pipeline for generating tsfresh derived features only. Input continues off the last node of feature_engineering_pipeline_instance
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
    )

    # Pipeline for generating lag features pipeline. No dependency on any past nodes in feature engineering but based on proxy revenue data partitioned into outlets in 02_dataloader
    lag_generation_pipeline = pipeline(
        [
            node(
                func=generate_lag,
                inputs=["loaded_proxy_revenue_partitioned_data", "parameters"],
                outputs="lag_features_partitions_dict",
                name="feature_engr_generate_lag",
            ),
        ],
    )

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
                name="feature_engr_merge_mmm_features_with_fold_outlets_training",
            ),
            node(
                func=merge_mmm_features_with_fold_outlets,
                inputs=[
                    "feature_engineering_validation",
                    "lightweightmmm_features_validation",
                ],
                outputs="merged_lightweightmmm_validation",
                name="feature_engr_merge_mmm_features_with_fold_outlets_validation",
            ),
            # Tsfresh feature merging
            node(
                func=merge_tsfresh_features_with_outlets,
                inputs=[
                    "merged_lightweightmmm_training",
                    "tsfresh_features_training",  # From feature_engineering
                ],
                outputs="merged_tsfresh_features_training",
                name="feature_engr_merge_tsfresh_features_with_fold_outlets_training",
            ),
            node(
                func=merge_tsfresh_features_with_outlets,
                inputs=[
                    "merged_lightweightmmm_validation",
                    "tsfresh_features_validation",  # From feature_engineering
                ],
                outputs="merged_tsfresh_features_validation",
                name="feature_engr_merge_tsfresh_features_with_fold_outlets_validation",
            ),
            # Lag features merging
            node(
                func=merge_fold_and_generated_lag,
                inputs=[
                    "merged_tsfresh_features_training",
                    "lag_features_partitions_dict",  # From feature_engineering
                ],
                outputs="merged_lag_features_training",
                name="feature_engr_merge_fold_and_generated_lag_training",
            ),
            node(
                func=merge_fold_and_generated_lag,
                inputs=[
                    "merged_tsfresh_features_validation",
                    "lag_features_partitions_dict",  # From feature_engineering
                ],
                outputs="merged_lag_features_validation",
                name="feature_engr_merge_fold_and_generated_lag_validation",
            ),
            # Concatenate same data folds
            node(
                func=concat_same_folds_of_outlet_data,
                inputs="merged_lag_features_training",
                outputs="concatenated_folds_training",
                name="feature_engr_preprocess_concat_same_folds_of_outlet_data_train",
            ),
            node(
                func=concat_same_folds_of_outlet_data,
                inputs="merged_lag_features_validation",
                outputs="concatenated_folds_validation",
                name="feature_engr_preprocess_concat_same_folds_of_outlet_data_val",
            ),
        ],
    )

    # Ordinal encoding pipeline
    ordinal_encoding_pipeline = pipeline(
        [
            node(
                func=apply_feature_encoding_fit,
                inputs=[
                    "concatenated_folds_training",
                    "parameters",
                ],
                outputs="feature_encoding_dict",
                name="feature_engr_apply_feature_encoding_fit",
            ),
            node(
                func=apply_feature_encoding_transform,
                inputs=[
                    "concatenated_folds_training",
                    "parameters",
                    "feature_encoding_dict",
                ],
                outputs="merged_features_training",
                name="feature_engr_apply_feature_encoding_transform_train",
            ),
            node(
                func=apply_feature_encoding_transform,
                inputs=[
                    "concatenated_folds_validation",
                    "parameters",
                    "feature_encoding_dict",
                ],
                outputs="merged_features_validation",
                name="feature_engr_apply_feature_encoding_transform_val",
            ),
        ]
    )

    # Extend pipeline based on togglable configuration enabling/disabling lightweightmmm and tsfresh.
    if conf_params["include_lightweightMMM"]:
        feature_engineering_pipeline_instance = (
            feature_engineering_pipeline_instance + lightweight_mmm_pipeline
        )

    if conf_params["include_tsfresh"]:
        feature_engineering_pipeline_instance = (
            feature_engineering_pipeline_instance + tsfresh_pipeline
        )

    # Extend existing pipeline with lag generation pipeline, merge generated features and ordinal encoding pipeline. Reason for putting ordinal encoding pipeline as the last is due to the fact that existing categorical data features are applicable across outlets and not specific to unique outlets.
    return (
        feature_engineering_pipeline_instance
        + lag_generation_pipeline
        + merge_generated_features_pipeline
        + ordinal_encoding_pipeline
    )
