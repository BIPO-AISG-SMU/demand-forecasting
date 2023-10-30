from typing import Any, Dict
import os
from kedro import pipeline
from kedro.framework.hooks import hook_impl
from kedro.io import DataCatalog, MemoryDataSet
from kedro.config import ConfigLoader
from kedro.framework.project import settings
from kedro.pipeline.node import Node
import logging
from shutil import rmtree
import glob
from pathlib import Path


class DataCleanupHooks:
    """This class serves as a Kedro Hook to execute necessary data cleanup procedure through the use of Kedro Hooks function declaration."""

    def __init__(self, LOGGER_NAME: str):
        self.logger = logging.getLogger(LOGGER_NAME)

    @hook_impl
    def before_pipeline_run(
        self,
        run_params: Dict[str, Any],
        pipeline: pipeline,
        catalog: DataCatalog,
    ):
        """This function adheres to kedro function declaration requirements by executing necessary data cleanup prior to any pipeline run(s). The cleanup affects existing data files that exists in both 1) path specified in catalog.yml as well as 2) the outputs generated from Kedro pipelines which are being used during execution.

        Args:
            run_params (Dict[str, Any]): Kedro run parameters in Dictionary.
            pipeline (pipeline): Kedro pipeline library utility.
            catalog (DataCatalog): Kedro catalog library utility.

        Raises:
            None.

        Returns:
            None.
        """
        # Coditional filtering to consider only paths specified in both catalog.yml and pipeline outputs which is executed.
        intermediate_files_dir = pipeline.all_outputs() & catalog._data_sets.keys()

        for dataset_name in intermediate_files_dir:
            try:
                filepath = Path(catalog._get_dataset(dataset_name)._path)
            except AttributeError:
                # For catalog requiring filepath parameter instead of path
                filepath = Path(catalog._get_dataset(dataset_name)._filepath)
            self.logger.info(f"Preparing removal of {filepath} if exists")
            try:
                if filepath.is_file():
                    os.remove(filepath)
                elif filepath.is_dir():
                    rmtree(filepath)
            except OSError:
                self.logger.error(
                    f"{filepath} cannot be removed due to permission issues or file does not exist."
                )
                continue
