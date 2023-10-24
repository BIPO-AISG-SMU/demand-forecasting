from kedro.pipeline import Pipeline, node, pipeline
import logging
from kedro.config import ConfigLoader
from bipo import settings

# import required nodes from training pipeline
from bipo.pipelines.data_loader.nodes import load_and_structure_marketing_data
from bipo.pipelines.time_agnostic_feature_engineering.feature_indicator_diff_creation import create_is_weekday_feature
from bipo.pipelines.feature_engineering.nodes import generate_lag
from bipo.pipelines.feature_engineering.encoding import one_hot_encoding_transform, ordinal_encoding_transform
from bipo.pipelines.feature_engineering.standardize_normalize import standard_norm_transform
from bipo_fastapi.nodes import *
from bipo.pipelines.feature_engineering.tsfresh_node import feature_engineering_by_outlet
from bipo.pipelines.feature_engineering.lightweightmmm import (
    generate_mmm_features,
)
from bipo.pipelines.time_agnostic_feature_engineering.nodes import (
    create_mkt_campaign_counts_start_end,
)

LOGGER = logging.getLogger(settings.LOGGER_NAME)
conf_loader = ConfigLoader(conf_source=settings.CONF_SOURCE)
conf_params = conf_loader["parameters"]
conf_const = conf_loader.get("constants*")
conf_inference = conf_loader.get("inference*")

def create_marketing_pipeline()->Pipeline:
    """Creates marketing pipeline which consists of feature engineering marketing features and lightweight_mmm features

    Returns:
        Pipeline: marketing pipeline
    """
    marketing_pipeline = Pipeline(
    [
        node(
            func=transform_mkt_input,
            inputs="mkt_df",
            outputs="new_mkt_df",
            name="preprare_mkt_data",
            tags="inference"
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
        #lightweight_mmm
        node(
            func=lambda lightweightmmm_params,std_norm_df: generate_mmm_features(lightweightmmm_params_artefact=lightweightmmm_params,outlet_df=std_norm_df,mmm_lags=conf_params["lightweightmmm_num_lags"],adstock_normalise=conf_params["lightweightmmm_adstock_normalise"],mkt_cost_col_list=conf_params["fe_mkt_channel_list"]),
            inputs=["lightweightmmm_params","mkt_output"],
            outputs="lightweightmmm_features",
            name="generate_lightweightmmm_features",
            tags="lightweight_mmm"
        ),
        node(
                func=merge_generated_features_to_main,
                inputs=["lightweightmmm_features","merged_mkt_df"],
                outputs="mmm_mkt_df",
                name="merge_generated_features_to_main_for_marketing",
                tags="lightweight_mmm"
            ),
    ]
)   
    # filter selected nodes by their tags to include/exclude lightweight_mmm
    if conf_inference["include_lightweight_mmm"]:
        return marketing_pipeline.only_nodes_with_tags("inference","lightweight_mmm")
    else: 
        return marketing_pipeline.only_nodes_with_tags("inference")

def create_lag_sales_pipeline()->Pipeline:
    """ creates the lag_sales pipeline which creates lag sales features and tsfresh features

    Returns:
        Pipeline: lag sales pipeline
    """
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
            func=lambda imputed_lag_sales_df: save_partition_dataset(df = imputed_lag_sales_df,partition_filepath = conf_const["inference"]["lag_sales_partition_filepath"]),
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
            inputs=["tsfresh_relevant_features", "lag_sales_partition","parameters"],
            outputs="tsfresh_features",
            name="feature_engineering_by_outlet",
            tags="tsfresh"
        ),
        node(
            func=merge_generated_features_to_main,
            inputs=["tsfresh_features", "final_lag_df"],
            outputs="merged_lag_sales_df",
            name="merge_generated_features_to_main_for_lag_sales",
            tags="tsfresh",
        ),
    ])
    # filter selected nodes by their tags to include/exclude lightweight_mmm
    if conf_inference["include_tsfresh"]:
        return lag_sales_pipeline.only_nodes_with_tags("inference","tsfresh")
    else: 
        return lag_sales_pipeline.only_nodes_with_tags("inference")

def create_outlet_pipeline()->Pipeline:
    """outlet pipeline which creates is_weekday_feature

    Returns:
        Pipeline: outlet pipeline
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

def create_merge_pipeline_outputs()->Pipeline:
    """merge outputs from marketing, lag_sales and outlet pipeline. Returns different pipelines based on the marketing and lag sales pipeline node outputs. This is dependent on whether to include tsfresh, lightweight_mmm or both or none.

    Returns:
        Pipeline: merge_pipeline_outputs pipeline
    """
    # includes both lightweight_mmm and tsfresh
    if conf_inference["include_lightweight_mmm"] and conf_inference["include_tsfresh"]:
        return Pipeline([node(
                func=merge_pipeline_output,
                inputs=["merged_lag_sales_df","mmm_mkt_df","processed_outlet_df"],
                outputs="merged_df",
                name="merge_tsfresh_and_lightweight_mmm",
            ),])
    # include lightweight_mmm, but not tsfresh
    elif conf_inference["include_lightweight_mmm"] and not conf_inference["include_tsfresh"]:
        return Pipeline([node(
                func=merge_pipeline_output,
                inputs=["final_lag_df","mmm_mkt_df","processed_outlet_df"],
                outputs="merged_df",
                name="merge_tsfresh_and_lightweight_mmm",
            ),])
    # include tsfresh but not lightweight_mmm
    elif conf_inference["include_tsfresh"] and not conf_inference["include_lightweight_mmm"]:
        return  Pipeline([node(
                func=merge_pipeline_output,
                inputs=["merged_lag_sales_df","merged_mkt_df","processed_outlet_df"],
                outputs="merged_df",
                name="merge_tsfresh_and_lightweight_mmm",
            ),])
    # no lightweight_mmm and no tsfresh
    else:
        return Pipeline([node(
                func=merge_pipeline_output,
                inputs=["final_lag_df","merged_mkt_df","processed_outlet_df"],
                outputs="merged_df",
                name="merge_tsfresh_and_lightweight_mmm",
            ),])

# feature engineering pipeline which includes encoding, standardizing/normalizing, tsfresh, lightweightmmm
def create_encoding_standardize_normalize_pipeline()->Pipeline:
    """Performs encoding and standardization/normalization, and processes final dataframe which will be used as the model input to make predictions. 

    Returns:
        Pipeline: encoding + standardization/normalization pipeline
    """
    # check if there is ordinal encoding and return corresponding pipeline
    ordinal_encoding_dict = conf_params["fe_ordinal_encoding_dict"]
    if ordinal_encoding_dict:
        return Pipeline(
        [   
            node(
            func=one_hot_encoding_transform,
                inputs=["merged_df","ohe_artefact"],
                outputs="one_hot_encoded_df",
                name="one_hot_encoding_transform",
                tags="inference"
            ),
            node(
            func=ordinal_encoding_transform,
                inputs=["one_hot_encoded_df","ordinal_encoding_artefact"],
                outputs="ordinal_encoded_df",
                name="ordinal_encoding_transform",
                tags="inference"
            ),
        node(
            func=standard_norm_transform,
            inputs=["ordinal_encoded_df","std_norm_artefact"],
            outputs="std_norm_df",
            name="standard_norm_transform",
            # tags="inference"
            ),
        node(
                func=process_final_merged_df,
                inputs="ordinal_encoded_df",
                outputs="processed_final_merged_df",
                name="process_final_merged_df_with_mmm",
                tags="inference"
            ),
        ])
    else:
        return Pipeline(
        [   
        node(
            func=one_hot_encoding_transform,
                inputs=["merged_df","ohe_artefact"],
                outputs="one_hot_encoded_df",
                name="one_hot_encoding_transform",
                tags="inference"
            ),
        node(
            func=standard_norm_transform,
            inputs=["one_hot_encoded_df","std_norm_artefact"],
            outputs="std_norm_df",
            name="standard_norm_transform",
            tags="inference"
            ),
        node(
                func=process_final_merged_df,
                inputs="one_hot_encoded_df",
                outputs="processed_final_merged_df",
                name="process_final_merged_df_with_mmm",
                tags="inference"
            ),
        ])

# merge all pipelines into 1 main pipeline
inference_pipeline = [
    create_marketing_pipeline(), 
    create_lag_sales_pipeline(),
    create_outlet_pipeline(),
    create_merge_pipeline_outputs(),create_encoding_standardize_normalize_pipeline()]
main_pipeline = pipeline([])
for individual_pipeline in inference_pipeline:
    main_pipeline += pipeline(
                    individual_pipeline,
                    inputs=None,
                    outputs=None,
                )
