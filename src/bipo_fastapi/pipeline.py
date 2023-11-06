from kedro.pipeline import Pipeline, node, pipeline
import logging
from kedro.config import ConfigLoader
from bipo import settings

# Import required nodes from training pipeline
from bipo.pipelines.data_loader.nodes import load_and_structure_marketing_data
from bipo.pipelines.time_agnostic_feature_engineering.feature_indicator_diff_creation import (
    create_is_weekday_feature,
)
from bipo.pipelines.feature_engineering.nodes import generate_lag
from bipo.pipelines.feature_engineering.encoding import (
    one_hot_encoding_transform,
    ordinal_encoding_transform,
)
from bipo.pipelines.feature_engineering.standardize_normalize import (
    standard_norm_transform,
)
from bipo_fastapi.nodes import *
from bipo_fastapi.load_data_catalog import load_data_catalog
from bipo.pipelines.feature_engineering.tsfresh_node import (
    feature_engineering_by_outlet,
)
from bipo.pipelines.time_agnostic_feature_engineering.nodes import (
    create_mkt_campaign_counts_start_end,
    generate_adstock,
)

LOGGER = logging.getLogger(settings.LOGGER_NAME)
CONF_LOADER = ConfigLoader(conf_source=settings.CONF_SOURCE)
CONF_INFERENCE = CONF_LOADER.get("inference*")


def create_marketing_pipeline() -> Pipeline:
    """Creates marketing pipeline which consists of feature engineering marketing features

    Returns:
        Pipeline: Marketing pipeline.
    """
    conf_params = CONF_LOADER["parameters"]
    marketing_pipeline = Pipeline(
        [
            node(
                func=transform_mkt_input,
                inputs="mkt_df",
                outputs="new_mkt_df",
                name="preprare_mkt_data",
                tags="inference",
            ),
            node(
                func=load_and_structure_marketing_data,
                inputs="new_mkt_df",
                outputs="mkt_output",
                name="transform_mkt",
                tags="inference",
            ),
            node(
                func=convert_to_string,
                inputs="mkt_output",
                outputs="transformed_name_mkt_df",
                name="convert_to_string",
                tags="inference",
            ),
            node(
                func=create_mkt_campaign_counts_start_end,
                inputs=["transformed_name_mkt_df", "parameters"],
                outputs="processed_mkt",
                name="process_mkt_campaign",
                tags=["inference", "modify_params"],
            ),
            node(
                func=process_marketing_df,
                inputs=["processed_mkt", "mkt_output"],
                outputs="merged_mkt_df",
                name="add_index_name",
                tags="inference",
            ),
            # adstock
            node(
                func=generate_adstock,
                inputs=["transformed_name_mkt_df", "parameters"],
                outputs="adstock_features",
                name="generate_adstock",
                tags="adstock",
            ),
            node(
                func=merge_adstock_and_marketing_df,
                inputs=["adstock_features", "merged_mkt_df"],
                outputs="merged_mkt_df_with_adstock",
                name="merge_adstock_and_marketing_df",
                tags="adstock",
            ),
        ]
    )
    # filter selected nodes by their tags to include/exclude adstock features
    if conf_params["include_adstock"]:
        return marketing_pipeline.only_nodes_with_tags("inference", "adstock")
    else:
        return marketing_pipeline.only_nodes_with_tags("inference")


def create_lag_sales_pipeline() -> Pipeline:
    """creates the lag_sales pipeline which creates lag sales features and tsfresh features

    Returns:
        Pipeline: lag sales pipeline.
    """
    conf_const = CONF_LOADER.get("constants*")
    lag_sales_pipeline = Pipeline(
        [
            node(
                func=impute_lag_sales_df,
                inputs=["lag_sales_df", "outlet_df"],
                outputs="imputed_lag_sales_df",
                name="impute_lag_sales",
                tags="inference",
            ),
            node(
                func=lambda imputed_lag_sales_df: save_partition_dataset(
                    df=imputed_lag_sales_df,
                    partition_filepath=conf_const["inference"][
                        "lag_sales_partition_filepath"
                    ],
                ),
                inputs="imputed_lag_sales_df",
                outputs="lag_sales_partition",
                name="save_partition_lag_sales",
                tags="inference",
            ),
            node(
                func=generate_lag,
                inputs=["lag_sales_partition", "parameters"],
                outputs="final_lag_df",
                name="generate_lag",
                tags="inference",
            ),
            node(
                func=feature_engineering_by_outlet,
                inputs=[
                    "tsfresh_relevant_features",
                    "lag_sales_partition",
                    "parameters",
                ],
                outputs="tsfresh_features",
                name="feature_engineering_by_outlet",
                tags="tsfresh",
            ),
            node(
                func=merge_generated_features_to_main,
                inputs=["tsfresh_features", "final_lag_df"],
                outputs="merged_lag_sales_df",
                name="merge_generated_features_to_main_for_lag_sales",
                tags="tsfresh",
            ),
        ]
    )
    # Filter selected nodes by their tags to include/exclude lightweight_mmm
    if CONF_INFERENCE["include_tsfresh"]:
        return lag_sales_pipeline.only_nodes_with_tags("inference", "tsfresh")
    else:
        return lag_sales_pipeline.only_nodes_with_tags("inference")


def create_outlet_pipeline() -> Pipeline:
    """outlet pipeline which creates is_weekday_feature.

    Returns:
        Pipeline: outlet pipeline.
    """
    return Pipeline(
        [
            node(
                func=create_is_weekday_feature,
                inputs="outlet_df",
                outputs="processed_outlet_df",
                name="create_is_weekday",
                tags="inference",
            ),
        ]
    )


def create_merge_pipeline_outputs() -> Pipeline:
    """merge outputs from marketing, lag_sales and outlet pipeline. Returns different pipelines based on the marketing and lag sales pipeline node outputs. This is dependent on whether to include tsfresh, adstock or both or none.

    Returns:
        Pipeline: merge_pipeline_outputs pipeline.
    """
    # includes adstock and tsfresh
    if conf_inference["include_tsfresh"] and conf_params["include_adstock"]:
        return Pipeline(
            [
                node(
                    func=merge_pipeline_output,
                    inputs=[
                        "merged_lag_sales_df",
                        "merged_mkt_df_with_adstock",
                        "processed_outlet_df",
                    ],
                    outputs="merged_df",
                    name="merge_tsfresh_and_adstock",
                ),
            ]
        )
    # include adstock, but not tsfresh
    elif conf_params["include_adstock"] and not conf_inference["include_tsfresh"]:
        return Pipeline(
            [
                node(
                    func=merge_pipeline_output,
                    inputs=[
                        "final_lag_df",
                        "merged_mkt_df_with_adstock",
                        "processed_outlet_df",
                    ],
                    outputs="merged_df",
                    name="merge_tsfresh_and_adstock",
                ),
            ]
        )
    # include tsfresh but not adstock
    elif conf_inference["include_tsfresh"] and not conf_params["include_adstock"]:
        return Pipeline(
            [
                node(
                    func=merge_pipeline_output,
                    inputs=[
                        "merged_lag_sales_df",
                        "merged_mkt_df",
                        "processed_outlet_df",
                    ],
                    outputs="merged_df",
                    name="merge_tsfresh_and_adstock",
                ),
            ]
        )
    # no adstock and no tsfresh
    else:
        return Pipeline(
            [
                node(
                    func=merge_pipeline_output,
                    inputs=["final_lag_df", "merged_mkt_df", "processed_outlet_df"],
                    outputs="merged_df",
                    name="merge_tsfresh_and_adstock",
                ),
            ]
        )


# feature engineering pipeline which includes encoding, standardizing/normalizing, tsfresh
def create_encoding_standardize_normalize_pipeline() -> Pipeline:
    """Performs encoding and standardization/normalization, and processes final dataframe which will be used as the model input to make predictions. One-hot encoding is compulsory, while ordinal encoding, standardization/normalization are optional.

    Returns:
        Pipeline: encoding + standardization/normalization pipeline.
    """
    catalog = load_data_catalog()
    ordinal_encoding_artefact = catalog.load("ordinal_encoding_artefact")
    standard_norm_artefact = catalog.load("std_norm_artefact")
    # One-hot encoding pipeline is a common pipeline for all pipeline combinations.
    ohe_pipeline = Pipeline(
        [
            node(
                func=one_hot_encoding_transform,
                inputs=["merged_df", "ohe_artefact"],
                outputs="one_hot_encoded_df",
                name="one_hot_encoding_transform",
                tags="inference",
            )
        ]
    )
    # Depending on if there is ordinal encoding, and standardization/normalization, create the corresponding pipelines.
    # Both ordinal encoding, and standardization/normalization
    if ordinal_encoding_artefact and standard_norm_artefact:
        LOGGER.info("Performing one-hot encoding, ordinal encoding, standard_norm")
        pipeline_instance = Pipeline(
            [
                node(
                    func=ordinal_encoding_transform,
                    inputs=["one_hot_encoded_df", "ordinal_encoding_artefact"],
                    outputs="ordinal_encoded_df",
                    name="ordinal_encoding_transform",
                    tags="ordinal_encoding",
                ),
                node(
                    func=standard_norm_transform,
                    inputs=["ordinal_encoded_df", "std_norm_artefact"],
                    outputs="std_norm_df",
                    name="standard_norm_transform",
                    tags="standard_norm",
                ),
                node(
                    func=process_final_merged_df,
                    inputs="std_norm_df",
                    outputs="processed_final_merged_df",
                    name="process_final_merged_df",
                    tags=["inference", "process_final_merged_df"],
                ),
            ]
        )
    # Ordinal encoding but no standardization/normalization
    elif ordinal_encoding_artefact and not standard_norm_artefact:
        LOGGER.info("Performing one-hot encoding, ordinal encoding")
        pipeline_instance = Pipeline(
            [
                node(
                    func=ordinal_encoding_transform,
                    inputs=["one_hot_encoded_df", "ordinal_encoding_artefact"],
                    outputs="ordinal_encoded_df",
                    name="ordinal_encoding_transform",
                    tags="ordinal_encoding",
                ),
                node(
                    func=process_final_merged_df,
                    inputs="ordinal_encoded_df",
                    outputs="processed_final_merged_df",
                    name="process_final_merged_df",
                    tags=["inference", "process_final_merged_df"],
                ),
            ]
        )
    # Standardization/normalization but no ordinal encoding
    elif standard_norm_artefact and not ordinal_encoding_artefact:
        LOGGER.info("Performing one-hot encoding, standard_norm")
        pipeline_instance = Pipeline(
            [
                node(
                    func=standard_norm_transform,
                    inputs=["one_hot_encoded_df", "std_norm_artefact"],
                    outputs="std_norm_df",
                    name="standard_norm_transform",
                    tags="standard_norm",
                ),
                node(
                    func=process_final_merged_df,
                    inputs="std_norm_df",
                    outputs="processed_final_merged_df",
                    name="process_final_merged_df",
                    tags=["inference", "process_final_merged_df"],
                ),
            ]
        )
    # No ordinal encoding, no standardization/normalization
    else:
        LOGGER.info("Performing one-hot encoding")
        pipeline_instance = Pipeline(
            [
                node(
                    func=process_final_merged_df,
                    inputs="std_norm_df",
                    outputs="processed_final_merged_df",
                    name="process_final_merged_df",
                    tags=["inference", "process_final_merged_df"],
                ),
            ]
        )
    # Append the common one-hot encoding pipeline
    return ohe_pipeline + pipeline_instance


# Merge all pipelines into 1 main pipeline
# Independent pipeline with separate inputs
independent_pipeline = pipeline(
    create_marketing_pipeline() + create_lag_sales_pipeline() + create_outlet_pipeline()
)

# Dependent pipeline which merges all the outputs from independent pipeline and performs encoding and standardization/normalization.
dependent_pipeline = pipeline(
    create_merge_pipeline_outputs() + create_encoding_standardize_normalize_pipeline()
)

main_pipeline = independent_pipeline + dependent_pipeline
