"""
This is a boilerplate pipeline 'data_preprocessing'
generated using Kedro 0.18.10
"""
from ...utils import get_logger
from .pipeline import create_pipeline
import sys

# Make sure old cache is not used
sys.dont_write_bytecode = True

# Inititate get_logging for data_processing. This will allow kedro logging to save the logs in the log folder
get_logger()

__all__ = ["create_pipeline"]

__version__ = "0.1"
