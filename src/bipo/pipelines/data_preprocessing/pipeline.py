"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.18.10
"""
from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline

from kedro.config import ConfigLoader
from kedro.framework.session import KedroSession

from .nodes import merge_outlet_and_other_df_feature, merge_non_proxy_revenue_data


def create_pipeline(**kwargs) -> Pipeline:
    pipeline_instance = pipeline(
        [
            node(
                func=merge_non_proxy_revenue_data,
                inputs="loaded_non_proxy_revenue_partitioned_data",
                outputs="merged_non_revenue_data",
                name="data_preprocessing_merge_non_proxy_revenue_data",
            ),
            node(
                func=merge_outlet_and_other_df_feature,
                inputs=[
                    "loaded_proxy_revenue_partitioned_data",
                    "merged_non_revenue_data",
                    "parameters",
                ],
                outputs="data_preprocessed",
                name="data_preprocessing_merge_outlet_and_other_features",
            ),
        ],
    )

    return pipeline_instance
