"""
This is a boilerplate pipeline 'data_split'
generated using Kedro 0.18.10
"""

import sys
from pathlib import Path

sys.path.append("../../")
sys.dont_write_bytecode = True

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import load_feature_engineered_data, do_train_val_test_split

# Recursively scan directories (config paths) contained in conf_source for configuration files with a yaml, yml, json, ini, pickle, xml or properties extension, load them, and return them in the form of a config dictionary.
from kedro.config import ConfigLoader
from kedro.framework.session import KedroSession
from kedro.framework.project import settings
from ...utils import get_project_root, get_logger

# This will set the log_modifier to create logs based on module affected
logging = get_logger()

ROOT_DIR = get_project_root()

# https://docs.kedro.org/en/stable/kedro.config.ConfigLoader.html#kedro.config.ConfigLoader. Point config path to project root directory
conf_path = str(ROOT_DIR / settings.CONF_SOURCE)
conf_loader = ConfigLoader(conf_source=conf_path)

# Get parameters key from config loader
conf_params = conf_loader["parameters"]

print(f"Conf params: {conf_params}")
conf_data_split = conf_params["data_split"]

print(f"Conf data split: {conf_data_split}")

OUTLET_LIST = conf_params["outlet_name"]


def create_pipeline(**kwargs) -> Pipeline:
    # Instantiate empty Pipeline to chain all Pipelines
    pipes = Pipeline([])

    outlets_list = ["AZ", "Z"]

    # Refer to catalog datasets naming
    for outlet in OUTLET_LIST:
        logging.info(f"Creating Pipeline for outlet {outlet}")
        outlet_catalog_reference = "_".join([outlet, "engineered"])

        # Define another separate Pipeline chaining data loading and splitting functions
        pipes += Pipeline(
            [
                # Node input/outputs should reference to catalog.yml identifier
                node(
                    func=load_feature_engineered_data,  # In nodes.py
                    inputs=f"{outlet_catalog_reference}",
                    outputs=f"loaded_processed_data_{outlet}",  # Interim data for next load, throw away after execution
                    name=f"node_load_data_{outlet}",
                ),
                # Indexing split
                node(
                    func=do_train_val_test_split,  # In nodes.py
                    inputs={
                        "df": f"loaded_processed_data_{outlet}",  # Gets the output of previous node
                        "config_dict": "params:data_split",  # Reads parameters.yml's data_split key
                    },
                    outputs=[f"{outlet}_train", f"{outlet}_val", f"{outlet}_test"],
                    name=f"node_split_data_{outlet}",
                ),
            ]
        )

    return pipes
