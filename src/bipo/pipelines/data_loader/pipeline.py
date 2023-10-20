from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline

# Import from nodes py
from .nodes import (
    merge_unique_csv_xlsx_df,
    merge_unique_daily_partitions,
    load_and_partition_proxy_revenue_data,
    load_and_structure_propensity_data,
    load_and_structure_marketing_data,
    load_and_structure_weather_data,
    rename_merge_unique_csv_xlsx_df_col_index,
)


def create_pipeline(**kwargs) -> Pipeline:
    # Custom node function for each different data
    non_unique_daily_records_pipeline_instance = pipeline(
        [
            node(
                func=load_and_structure_propensity_data,
                inputs="raw_propensity_data",
                outputs="loaded_propensity_data",
                name="data_loader_propensity_data",
            ),
            node(
                func=load_and_structure_weather_data,
                inputs="raw_weather_data",
                outputs="loaded_weather_data",
                name="data_loader_weather_data",
            ),
            node(
                func=load_and_structure_marketing_data,
                inputs="raw_marketing_data",
                outputs="loaded_marketing_data",
                name="data_loader_marketing_data",
            ),
            node(
                func=load_and_partition_proxy_revenue_data,
                inputs="raw_proxy_revenue_data",
                outputs="loaded_proxy_revenue_partitioned_data",
                name="data_loader_proxy_revenue_data",
            ),
        ],
    )
    # Handles unique daily records file
    unique_daily_records_pipeline_instance = pipeline(
        [
            node(
                func=merge_unique_daily_partitions,
                inputs=["xlsx_raw_unique_daily_partitions"],
                outputs="merged_xlsx",  # MemoryDataset
                name="data_loader_merge_xlsx",
            ),
            node(
                func=merge_unique_daily_partitions,
                inputs=["csv_raw_unique_daily_partitions"],
                outputs="merged_csv",  # MemoryDataset
                name="data_loader_merge_csv",
            ),
            node(
                func=merge_unique_csv_xlsx_df,
                inputs=["merged_csv", "merged_xlsx"],
                outputs="merged_unique_daily_temp",
                name="data_loader_merge_csv_and_xlsx",
            ),
            node(
                func=rename_merge_unique_csv_xlsx_df_col_index,
                inputs="merged_unique_daily_temp",
                outputs="merged_unique_daily",
                name="data_loader_rename_column_index",
            ),
        ]
    )

    # Combine all pipelines
    return (
        unique_daily_records_pipeline_instance
        + non_unique_daily_records_pipeline_instance
    )
