"""
This is a boilerplate pipeline 'model_training'
generated using Kedro 0.18.11
"""
import sys

sys.dont_write_bytecode = False

from .pipeline import create_pipeline

__all__ = ["create_pipeline"]

__version__ = "0.1"
