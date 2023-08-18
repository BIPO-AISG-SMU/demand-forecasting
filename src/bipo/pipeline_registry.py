"""Project pipelines."""
from typing import Dict

# Library import
from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline, pipeline

# Our pipelines subdirectory
from bipo.pipelines import data_loader as dloader

from bipo.pipelines import data_preprocessing as dpreprocess

from bipo.pipelines import feature_engineering as fe

# from bipo.pipelines import data_split as dsplit

# This will get the active logger during run time


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines. Currently only imports and registers data split pipeline into pipeline registry.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    # Register data_loader create pipeline
    data_loader_pipeline = dloader.create_pipeline()

    # Register data_preprocessing create pipeline
    data_preprocessing_pipeline = dpreprocess.create_pipeline()

    # Register feature engineering create pipeline
    feature_engineering_pipeline = fe.create_feature_engineering_pipeline()
    feature_selection_pipeline = fe.create_feature_selection_pipeline()

    # Register datasplit create pipeline
    # data_split_pipeline = dsplit.create_pipeline()

    # __default__ line in the return statement indicates the default sequence of modular pipelines to run

    # Ideal state
    return_dict = {
        "dloader": data_loader_pipeline,
        "data_preprocessing": data_preprocessing_pipeline,
        "feature_engineering": feature_engineering_pipeline,
        "feature_selection": feature_selection_pipeline,
        "__default__": data_preprocessing_pipeline,
    }  # + data_split_pipeline, "dsplit": data_split_pipeline,

    return return_dict
