"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.18.10
"""
# conda activate bipo_dev
# kedro run -p data_processing

# Standard Library Imports
from pathlib import Path
import pandas as pd
import os

# Third Party Imports
from kedro.io import DataCatalog
from kedro_datasets.pandas import CSVDataSet
from kedro.config import ConfigLoader
from kedro.pipeline import Pipeline, node
import kedro

# Local Application/Custom Imports
from bipo.pipelines.data_preprocessing.preprocessing_node import DataPreprocessing

# This will set the log_modifier to create logs based on module affected
from bipo.utils import (
    get_project_path,
    get_input_output_folder,
    add_dataset_to_catalog,
)  # , get_logger
import logging

# logging = logging.getLogger("kedro")
logging = logging.getLogger(__name__)
# Create an empty data catalog
data_catalog = DataCatalog()

# initiate to constants.yml and parameters.yml
project_path = get_project_path()
conf_loader = ConfigLoader(conf_source=project_path / "conf")
conf_params = conf_loader["parameters"]
constants = conf_loader.get("constants*")["data_preprocessing"]

# This will get dataloader_data from catalog.yml
conf_catalog = conf_loader.get("catalog*", "catalog/**")
catalog = DataCatalog.from_config(conf_catalog)

# load configs
SOURCE_DATA_DIR = constants["preprocessing_data_source_dir"]
DEST_DATA_DIR = constants["preprocessing_data_destination_dir"]


def add_node_to_pipeline(input_df: pd.DataFrame, output_filename: str) -> Pipeline:
    """Create a pipeline and adds a node to it.

    Args:
        input_filename (str): The filename of the input data.
        output_filename (str): The filename for the output data.

    Returns:
        Pipeline: The kedro pipeline with the added node.
    """
    dp = DataPreprocessing()

    return Pipeline(
        [
            node(
                func=lambda: dp.run_pipeline(input_df),
                inputs=None,
                outputs=output_filename,
            ),
            node(
                func=lambda output_dataset, output_filename=output_filename: data_catalog.save(
                    output_filename, output_dataset
                ),
                inputs=output_filename,
                outputs=None,
            ),
        ]
    )


def create_pipeline(**kwargs) -> Pipeline:
    """Creates a pipeline that processes all csv files in the input folder.

    Returns:
        Pipeline: The kedro pipeline for data preprocessing.
    """
    # Load the input and output folders
    input_folder, output_folder = get_input_output_folder(
        SOURCE_DATA_DIR,
        DEST_DATA_DIR,
    )

    # Create an empty chain of pipeline
    pipeline_chain = Pipeline([])

    # Terminate program by returning if directory does not exist
    if not os.path.exists(input_folder):
        log_string = f"{input_folder} not found. Please check config. Exiting...."
        logging.error(log_string)
    else:
        # List contents in the directory
        logging.info(f"Processing contents in: {input_folder}")
        # load partitioned datasets
        partitioned_input = catalog.load("dataloader_data")
        for partition_id, partition_load_func in sorted(partitioned_input.items()):
            if "merged_" in partition_id:
                partition_id = partition_id.split(".csv")[0]

                # like pd.read_csv. partition_data = input_df
                partition_data = partition_load_func()

                # Add output save filepath to data_catalog
                output_filename = f"{partition_id}_processed"
                output_file_path = output_folder / (output_filename + ".csv")
                add_dataset_to_catalog(data_catalog, output_filename, output_file_path)

                # Add individual pipeline to main Pipeline
                out_pipeline = add_node_to_pipeline(partition_data, output_filename)

                # Append the pipeline
                pipeline_chain += out_pipeline

    return pipeline_chain
