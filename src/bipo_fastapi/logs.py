import logging
import yaml
from pathlib import Path

# Define a logger for this module
LOGGER = logging.getLogger(__name__)


def setup_logging(logging_config_path: str, default_level: int = logging.INFO):
    """
    load logging configuration and setup logger.

    Args:
        logging_config_path (str): Path to the YAML file containing the logging configuration.
        default_level (int, optional): Default logging level to use if the configuration file
                                        cannot be loaded. Defaults to logging.INFO.
    """
    try:
        with open(logging_config_path, "rt") as file:
            log_config = yaml.safe_load(file.read())

        logging.config.dictConfig(log_config)

    except FileNotFoundError:
        # If the configuration file cannot be found, fall back to a basic logging configuration.
        logging.basicConfig(
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            level=default_level,
        )
        LOGGER.info(
            f"Logging config file {logging_config_path} is not found. Basic config is being used."
        )

    except yaml.YAMLError as e:
        # If there's an error parsing the configuration file, fall back to a basic logging configuration.
        logging.basicConfig(
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            level=default_level,
        )
        LOGGER.info(
            f"Error parsing logging config file: {e}. Basic config is being used."
        )
