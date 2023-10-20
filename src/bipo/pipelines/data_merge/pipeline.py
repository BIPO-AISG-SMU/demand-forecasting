from kedro.pipeline import Pipeline, node, pipeline
from .nodes import concat_outlet_preprocessed_data

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=concat_outlet_preprocessed_data,
                inputs="data_preprocessed",
                outputs="data_merge",
                name="data_merge_concat_outlet_preprocessed_data",
            ),
        ],
    )