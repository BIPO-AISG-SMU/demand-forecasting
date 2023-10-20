"""
This is a boilerplate pipeline 'pre_feature_engineering'
generated using Kedro 0.18.10
"""

from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline

from .nodes import (
    segregate_outlet_based_train_val_test_folds,
    create_bool_feature_and_differencing,
    no_mkt_days_imputation,
    drop_columns,
    create_mkt_campaign_counts_start_end,
    merge_fold_partitions_and_gen_mkt_data,
)
from bipo import settings
from kedro.config import ConfigLoader
conf_loader = ConfigLoader(conf_source=settings.CONF_SOURCE)

def create_pipeline(**kwargs) -> Pipeline:
    data_split_source = conf_loader["parameters"]["split_approach_source"]
    pipeline_instance = pipeline(
        [
            node(
                func=create_mkt_campaign_counts_start_end,
                inputs=["loaded_marketing_data", "parameters"],
                outputs="mkt_indicator_features",
                name="time_agnostic_fe_create_mkt_campaign_counts_start_end",
            ),
            node(
                func=merge_fold_partitions_and_gen_mkt_data,
                inputs=[f"{data_split_source}.data_split", "mkt_indicator_features"],
                outputs="merged_folds_mkt",
                name="time_agnostic_fe_merge_fold_partitions_and_gen_mkt_data",
            ),
            node(
                func=no_mkt_days_imputation,
                inputs=["merged_folds_mkt", "parameters"],
                outputs="no_mkt_imputation_data",
                name="time_agnostic_fe_no_mkt_days_imputation",
            ),
            node(
                func=create_bool_feature_and_differencing,
                inputs=["no_mkt_imputation_data", "parameters"],
                outputs="bool_feature_and_diff_data",
                name="time_agnostic_fe_create_bool_feature_and_differencing",
            ),
            node(
                func=drop_columns,
                inputs=[
                    "bool_feature_and_diff_data",
                    "parameters",
                ],  # Uses output from create_bool_feature_and_differencing node
                outputs="dropped_columns_data",
                name="pre_drop_columns",
            ),
            node(
                func=segregate_outlet_based_train_val_test_folds,
                inputs=["dropped_columns_data", "parameters"],
                outputs=[
                    "time_agnostic_feature_engineering_training",
                    "time_agnostic_feature_engineering_validation",
                    "time_agnostic_feature_engineering_testing",
                ],
                name="time_agnostic_fe_segregate_outlet_based_train_val_test_folds",
            ),
        ],
    )

    return pipeline_instance
