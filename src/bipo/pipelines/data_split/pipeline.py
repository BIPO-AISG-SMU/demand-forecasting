"""
This is a boilerplate pipeline 'data_split'
generated using Kedro 0.18.10
"""
# Adjusted for namespace
from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline
from .nodes import (
    do_time_based_data_split,
    prepare_for_split,
)


def create_pipeline(**kwargs) -> Pipeline:
    # New pipeline for data split to add all subpipelines
    pipeline_instance = pipeline(
        [
            node(
                func=prepare_for_split,  # In nodes.py
                inputs=["data_merge", "params:data_split_params"],
                outputs="split_params_dict",
                name="data_split_prepare_for_split",
            ),
            node(
                func=do_time_based_data_split,  # In nodes.py
                inputs="split_params_dict",
                outputs="data_split",
                name="data_split_do_time_based_data_split",
            ),
        ],
    )

    # Instantiate multiple instances of pipelines with static structure, but dynamic inputs/outputs/parameters. Inputs/outputs required if not managed by namespace
    simple_split_pipeline = pipeline(
        pipe=pipeline_instance,
        inputs=["data_merge"],
        parameters={
            "params:data_split_params": "params:simple_split",
        },  # namespace mapped to data_split.yml namespace declaration
        namespace="simple_split",
    )

    expanding_window_pipeline = pipeline(
        pipe=pipeline_instance,
        inputs=["data_merge"],
        parameters={
            "params:data_split_params": "params:expanding_window",
        },  # namespace mapped to data_split.yml namespace declaration
        namespace="expanding_window",
    )

    sliding_window_pipeline = pipeline(
        pipe=pipeline_instance,
        inputs=["data_merge"],
        parameters={
            "params:data_split_params": "params:sliding_window",
        },  # namespace mapped to data_split.yml namespace declaration
        namespace="sliding_window",
    )

    return sliding_window_pipeline + expanding_window_pipeline + simple_split_pipeline
