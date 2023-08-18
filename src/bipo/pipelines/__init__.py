import sys

sys.dont_write_bytecode = True
from bipo.utils import get_logger, get_project_path

# This will set the log_modifier to create logs based on module affected
get_logger()
