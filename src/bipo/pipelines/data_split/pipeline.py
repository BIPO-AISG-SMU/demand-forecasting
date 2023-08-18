"""
This is a boilerplate pipeline 'data_split'
generated using Kedro 0.18.10
"""
import sys
from pathlib import Path

sys.path.append("../../")
sys.dont_write_bytecode = True

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import load_concat_outlet_feature_engineered_data, do_train_val_test_split

# Recursively scan directories (config paths) contained in conf_source for configuration files with a yaml, yml, json, ini, pickle, xml or properties extension, load them, and return them in the form of a config dictionary.
from kedro.config import ConfigLoader
from kedro.framework.session import KedroSession
from kedro.framework.project import settings

# This will set the log_modifier to create logs based on module affected

def create_pipeline(**kwargs) -> Pipeline:
    # Instantiate empty Pipeline to chain all Pipelines
    return pipeline(
        [
            node(
                func=load_concat_outlet_feature_engineered_data,
                inputs="feature_engineering_data",
                outputs="concat_outlet_dataset",  # MemoryDataSet
                name="load_featureengineered_training_data",
            ),
            node(
                func=do_train_val_test_split,  # In nodes.py
                inputs={
                    "df": "concat_outlet_dataset",  # Gets the output of previous MemoryDataset
                    "config_dict": "params:data_split",  # Reads parameters.yml's data_split key
                },
                outputs="train_val_test_data_split",
                # outputs=["training_data", "validation_data", "testing_data"],
                name="split_data",
            ),
        ]
    )
