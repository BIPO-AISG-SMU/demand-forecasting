"""This package contains modules pertaining to differing parts
of the end-to-end workflow, excluding the source for serving
the model through a REST API."""
from bipo.utils import get_logger

import sys

# Make sure old cache is not used
sys.dont_write_bytecode = True

# Inititate get_logging for data_processing. This will allow kedro logging to save the logs in the log folder
get_logger()
