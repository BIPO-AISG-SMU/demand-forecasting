"""Project settings. There is no need to edit this file unless you want to change values
from the Kedro defaults. For further information, including these default values, see
https://kedro.readthedocs.io/en/stable/kedro_project_setup/settings.html."""
import warnings

# Instantiated project hooks.
from bipo.hooks.DataCleanupHooks import DataCleanupHooks
from bipo.hooks.SetupDailyLogsHooks import SetupDailyLogsHooks
from bipo.hooks.MemoryProfilingHooks import MemoryProfilingHooks

from kedro.framework.context.context import KedroContext
from bipo.utils import get_project_path
import os

warnings.simplefilter("ignore", category=DeprecationWarning)

# Directory that holds configuration.
CONF_SOURCE = os.path.join(get_project_path(), "conf")
LOGGER_NAME = "kedro"

# hooks import
HOOKS = (
    DataCleanupHooks(LOGGER_NAME=LOGGER_NAME),
    SetupDailyLogsHooks(LOGGER_NAME=LOGGER_NAME),
    MemoryProfilingHooks(LOGGER_NAME=LOGGER_NAME),
)

# Installed plugins for which to disable hook auto-registration.
# DISABLE_HOOKS_FOR_PLUGINS = ("kedro-viz",)

# Class that manages storing KedroSession data.
# from kedro.framework.session.store import BaseSessionStore
# SESSION_STORE_CLASS = BaseSessionStore
# Keyword arguments to pass to the `SESSION_STORE_CLASS` constructor.
# SESSION_STORE_ARGS = {
#     "path": "./sessions"
# }

# # Class that manages how configuration is loaded.
# from kedro.config import ConfigLoader  # new import
# CONFIG_LOADER_CLASS = ConfigLoader

# # # Keyword arguments to pass to the `CONFIG_LOADER_CLASS` constructor.
# CONFIG_LOADER_ARGS = {
#     "config_patterns": {
#         "constants": ["constants*/, *constants/**"],
#         "logging": ["logging*/"]
#     }
# }

# Class that manages Kedro's library components.
# from kedro.framework.context import KedroContext
# CONTEXT_CLASS = KedroContext

# Class that manages the Data Catalog.
# from kedro.io import DataCatalog
# DATA_CATALOG_CLASS = DataCatalog
