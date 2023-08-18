"""
This is a boilerplate pipeline 'feature_engineering'
generated using Kedro 0.18.10
"""
import sys

sys.dont_write_bytecode = True
from .pipeline import (
    create_feature_engineering_pipeline,
    create_feature_selection_pipeline,
)
from bipo.utils import get_logger

get_logger()

# from .pipeline import create_pipeline

# __all__ = ["create_pipeline"]

__version__ = "0.1"
