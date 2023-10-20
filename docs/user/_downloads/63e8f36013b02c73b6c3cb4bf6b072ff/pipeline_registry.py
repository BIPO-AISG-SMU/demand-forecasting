"""Project pipelines."""
from typing import Dict

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline, pipeline

# Library import
# Our pipelines subdirectory
from bipo.pipelines import (
    data_loader as dloader,
    data_preprocessing as dpreprocess,
    data_merge as dmerge,
    data_split as dsplit,
    time_agnostic_feature_engineering as time_agnostic_fe,
    feature_engineering as fe,
    model_specific_preprocessing as mpreprocess,
    model_training as mtrain,
    model_evaluation as meval,
)

def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines. Currently only imports and registers data split pipeline into pipeline registry.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    # Register data_loader create pipeline
    data_loader_pipeline = dloader.create_pipeline()

    # Register data_preprocessing create pipeline
    data_preprocessing_pipeline = dpreprocess.create_pipeline()

    # Register datasplit create pipeline
    data_merge_pipeline = dmerge.create_pipeline()
    data_split_pipeline = dsplit.create_pipeline()

    # Register feature engineering pipelines
    time_agnostic_feature_engineering_pipeline = time_agnostic_fe.create_pipeline()

    time_dependent_feature_engineering_pipeline = fe.create_pipeline()
    model_spec_preprocess_pipeline = mpreprocess.create_pipeline()

    # Register model train create pipeline
    model_train_pipeline = mtrain.create_pipeline()

    model_eval_pipeline = meval.create_pipeline()

    # __default__ line in the return statement indicates the default sequence of modular pipelines to run

    # Dictionary of pipeline. Note that no data_loader included as concurrency issues between data loader
    pipeline_registry_dict = {
        "data_loader": data_loader_pipeline,
        # For running data pipeline. Note "data_loader" need to be ran seperately.
        "data_pipeline": data_preprocessing_pipeline
        + data_merge_pipeline
        + data_split_pipeline
        + time_agnostic_feature_engineering_pipeline
        + time_dependent_feature_engineering_pipeline
        + model_spec_preprocess_pipeline,
        "training_pipeline": model_train_pipeline + model_eval_pipeline,
        "__default__": data_preprocessing_pipeline
        + data_merge_pipeline
        + data_split_pipeline
        + time_agnostic_feature_engineering_pipeline
        + time_dependent_feature_engineering_pipeline
        + model_spec_preprocess_pipeline
        + model_train_pipeline
        + model_eval_pipeline,
    }

    return pipeline_registry_dict
