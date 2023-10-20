from typing import Any, Dict
import os
from kedro import pipeline
from kedro.framework.hooks import hook_impl
from kedro.io import DataCatalog
from kedro.config import ConfigLoader
from kedro.framework.project import settings
from datetime import datetime
import logging
import warnings


class SetupDailyLogsHooks:
    """This class serves as a Kedro Hook to faciltate the creation/configuration of daily logs file with filenames ddmmyyyy_info.log before pipeline run."""

    def __init__(self, LOGGER_NAME):
        self._mem_usage = {}
        self.logger = logging.getLogger(LOGGER_NAME)

    @hook_impl
    def before_pipeline_run(
        self,
        run_params: Dict[str, Any],
        pipeline: pipeline,
        catalog: DataCatalog,
    ):
        """This function overwrites the logger path specified in logger.yml with a custom time-based ddmmmyyyyy_<log_level>.log specification, which includes info, debug and error log levels.

        Args:
            run_params (Dict[str, Any]): Kedro run parameters in Dictionary.
            pipeline (pipeline): Kedro pipeline library utility.
            catalog (DataCatalog): Kedro catalog library utility.

        Raises:
            None.

        Returns:
            None.
        """
        conf_loader = ConfigLoader(conf_source=settings.CONF_SOURCE)
        config_dict = conf_loader.get("logging*")

        # Based on logger.yml handlers.
        file_handler_list = [
            "info_file_handler",
            "debug_file_handler",
            "error_file_handler",
        ]

        for file_handler_type in file_handler_list:
            # Assumes of the format logs/info.log or similar structure
            # base, extension = os.path.splitext(filename)
            level_info_name = config_dict["handlers"][file_handler_type]["level"]
            # Get current run date in ddmmyyyy format
            today = datetime.today()
            today_str = today.strftime("%d/%m/%Y").replace("/", "")
            # Construct the required format

            # Make parent folder containing submodule name if avail, regardless of existence as long submodule_name.is not None
            log_dir_for_submodule = "logs"
            log_filename = f"{today_str}_{level_info_name.lower()}.log"

            orig_log_file = config_dict["handlers"][file_handler_type]["filename"]
            # Delete default configure log filepaths to prevent unnecessary confusion between generated logs
            if os.path.isfile(orig_log_file):
                os.remove(orig_log_file)
            # Construct relevant logfile name based on kedro logger structure and update config path
            rel_logfile_path = os.path.join(log_dir_for_submodule, log_filename)
            config_dict["handlers"][file_handler_type]["filename"] = rel_logfile_path

        # Apply the configuration change assuming it has config
        try:
            logging.config.dictConfig(config_dict)
        except KeyError:
            pass

        return None
