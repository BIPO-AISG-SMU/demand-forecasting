from .nodes import train_model

from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline
from kedro.config import ConfigLoader
from kedro.framework.session import KedroSession
from bipo import settings

conf_loader = ConfigLoader(conf_source=settings.CONF_SOURCE)
conf_params = conf_loader["parameters"]
conf_constants = conf_loader.get("constants*")


def create_pipeline(**kwargs) -> Pipeline:
    model = conf_loader["parameters"]["model"]  # Name of model
    valid_model_list = conf_constants["modeling"]["valid_model_name"]

    # Override invalid model name
    if model not in valid_model_list:
        model = conf_constants["modeling"]["model_name_default"]

    # Pipeline for model specific preprocessing. Inputs must be string
    pipeline_instance = pipeline(
        [
            node(
                func=train_model,  # In nodes.py
                inputs=[
                    "model_specific_preprocessing_train",
                    "parameters",
                    "params:model_params_dict",
                ],
                outputs="model_training_artefact",
                name="model_train_train_model",
                tags="model_training",
            ),
            node(
                func=explain_ebm,  # In nodes.py
                inputs=[
                    "model_training_artefact",
                    "model_specific_preprocessing_train",
                ],
                outputs=None,
                name="explain_ebm",
                tags="enable_explainability",
            ),
        ],
    )
    if conf_params["model"] == "ebm" and conf_params["enable_explainability"]:
        pipeline_instance = pipeline_instance.only_nodes_with_tags(
            "model_training", "enable_explainability"
        )
    else:
        pipeline_instance = pipeline_instance.only_nodes_with_tags("model_training")

    train_model_pipeline = pipeline(
        pipe=pipeline_instance,
        parameters={"params:model_params_dict": f"params:{model}"},
        namespace=model,  # Governs input/output to be namespaced controlled
    )

    return train_model_pipeline
