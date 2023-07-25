"""Project pipelines."""
from __future__ import annotations

from typing import Dict

from kedro.pipeline import Pipeline, pipeline

from bipo.pipelines import data_split as dsplit

# This will get the active logger during run time


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines. Currently only imports and registers data split pipeline into pipeline registry.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """

    # Register datasplit create pipeline
    data_split_pipeline = dsplit.create_pipeline()
    # __default__ line in the return statement indicates the default sequence of modular pipelines to run
    return {"dsplit": data_split_pipeline, "__default__": data_split_pipeline}
