"""
This is a boilerplate pipeline 'data_loader'
generated using Kedro 0.18.10
"""
import os
from kedro.pipeline import Pipeline, node, pipeline
from kedro.io import MemoryDataSet
from .dataloader import (
    merge_all_data,
    add_save_filepath_to_catalog,
    save_data,
)
from .data_check import DataCheck
from .read_proxyrevenue import ReadProxyRevenueData
from .read_propensity import ReadPropensityData
from .read_climate import ReadClimateData
from .read_covid import ReadCovidData
from .read_holiday import ReadHolidayData
from .read_marketing import ReadMarketingData
from .dataloader import check_outlet_location, check_number_of_zeros_proxy_revenue

from kedro.config import ConfigLoader
from kedro.framework.session import KedroSession
from kedro.framework.project import settings
from kedro.runner import SequentialRunner
import pandas as pd
from bipo.utils import get_project_path

# initiate to constants.yml and parameters.yml
project_path = get_project_path()
conf_loader = ConfigLoader(conf_source=project_path / "conf")
conf_params = conf_loader["parameters"]["dataloader"]
constants = conf_loader.get("constants*")["dataloader"]

# Read fix parameters from constants.yml
SOURCE_DATA_DIR = constants["data_source_dir"]
proxy_revenue_file = conf_params["proxy_revenue_file"]
propensity_file = conf_params["propensity_file"]
marketing_file = conf_params["marketing_file"]
climate_record_file = conf_params["climate_record_file"]
covid_record_file = conf_params["covid_record_file"]
holiday_data_file = conf_params["holiday_data_file"]

# Get all files in 01_raw folder
path_to_data = os.path.join(project_path, *SOURCE_DATA_DIR)

# pipeline will read in all 7 datasets and use the outputs as inputs for final merge_all_data node


def create_pipeline(**kwargs) -> Pipeline:
    # read proxy revenue
    proxy_revenue_file_path = os.path.join(path_to_data, proxy_revenue_file)
    read_proxy = ReadProxyRevenueData()
    proxy_revenue_df = read_proxy.read_data(proxy_revenue_file_path)

    ## location specific data
    # read propensity
    propensity_file_path = os.path.join(path_to_data, propensity_file)
    read_propensity = ReadPropensityData(proxy_revenue_df)

    # read climate
    climate_record_file_path = os.path.join(path_to_data, climate_record_file)
    read_climate = ReadClimateData(proxy_revenue_df)

    # read holiday
    holiday_data_file_path = os.path.join(path_to_data, holiday_data_file)
    read_holiday = ReadHolidayData()

    # read covid
    covid_record_file_path = os.path.join(path_to_data, covid_record_file)
    read_covid = ReadCovidData()

    # read marketing
    marketing_file_path = os.path.join(path_to_data, marketing_file)
    read_marketing = ReadMarketingData()

    # define the read pipeline
    read_data_pipeline = pipeline(
        [
            node(
                func=lambda propensity_file_path=propensity_file_path,: read_propensity.read_data(
                    propensity_file_path
                ),
                inputs=None,
                outputs="raw_propensity",
                name="raw_propensity_node",
            ),
            node(
                func=lambda climate_record_file_path=climate_record_file_path,: read_climate.read_data(
                    climate_record_file_path
                ),
                inputs=None,
                outputs="raw_climate",
                name="raw_climate_node",
            ),
            node(
                func=lambda holiday_data_file_path=holiday_data_file_path,: read_holiday.read_data(
                    holiday_data_file_path
                ),
                inputs=None,
                outputs="raw_holiday",
                name="raw_holiday_node",
            ),
            node(
                func=read_holiday.preprocess_data,
                inputs=["raw_holiday"],
                outputs="preprocessed_holiday_data",
                name="preprocessed_holiday_node",
            ),
            node(
                func=lambda covid_record_file_path=covid_record_file_path,: read_covid.read_data(
                    covid_record_file_path
                ),
                inputs=None,
                outputs="raw_covid",
                name="raw_covid_node",
            ),
            node(
                func=read_covid.preprocess_data,
                inputs=["raw_covid"],
                outputs="preprocessed_covid_data",
                name="preprocessed_covid_node",
            ),
            node(
                func=lambda marketing_file_path=marketing_file_path,: read_marketing.read_campaign_data(
                    marketing_file_path
                ),
                inputs=None,
                outputs="preprocessed_marketing_campaign_data",
                name="preprocessed_marketing_campaign_node",
            ),
            node(
                func=lambda marketing_file_path=marketing_file_path,: read_marketing.read_ad_data(
                    marketing_file_path
                ),
                inputs=None,
                outputs="preprocessed_marketing_ad_data",
                name="preprocessed_marketing_ad_node",
            ),
            node(
                func=read_marketing.merge_marketing,
                inputs=[
                    "preprocessed_marketing_campaign_data",
                    "preprocessed_marketing_ad_data",
                ],
                outputs="preprocessed_marketing_data",
                name="preprocessed_marketing_node",
            ),
        ]
    )
    # create auto_pipeline
    auto_pipeline = Pipeline([])

    # add read_data_pipeline to auto_pipeline
    auto_pipeline += read_data_pipeline

    # Get cost Centre code list from proxy_revenue_df
    COST_CENTRE_CODE_LIST = proxy_revenue_df["CostCentreCode"].unique().tolist()

    # from parameters.yml, specific cost centre code to remove
    SPECIFIC_COST_CENTRE_CODE_LIST = conf_params["exclude_cost_centre_code"]

    # Update the COST_CENTRE_CODE_LIST and remove any cost centre from COST_CENTRE_CODE_LIST seen in SPECIFIC_COST_CENTRE_CODE_LIST
    COST_CENTRE_CODE_LIST = list(
        set(COST_CENTRE_CODE_LIST) - set(SPECIFIC_COST_CENTRE_CODE_LIST)
    )

    for code in COST_CENTRE_CODE_LIST:
        # convert int to str
        code = str(code)
        # Check if cost centre has many 0 in proxy revenue
        code_with_many_zeros = check_number_of_zeros_proxy_revenue(
            proxy_revenue_df, code
        )
        if code_with_many_zeros == code:
            continue  # Skip this iteration and move to the next one
        else:
            # add missing datasets to catalog
            add_save_filepath_to_catalog(proxy_revenue_df, code)

            # define the preprocess pipeline
            data_loader = pipeline(
                [
                    # Node to look for cost center location
                    node(
                        func=lambda proxy_revenue_df=proxy_revenue_df, code=code,: check_outlet_location(
                            proxy_revenue_df, code
                        ),
                        inputs=None,
                        outputs="location",
                        name="check_outlet_location_node",
                    ),
                    node(
                        func=lambda proxy_revenue_df=proxy_revenue_df, code=code,: read_proxy.preprocess_data(
                            proxy_revenue_df, code
                        ),
                        inputs=None,
                        outputs="preprocessed_proxy_revenue_data",
                        name="preprocessed_proxy_revenue_node",
                    ),
                    node(
                        func=read_propensity.preprocess_data,
                        inputs=["raw_propensity", "location"],
                        outputs="preprocessed_propensity_data",
                        name="preprocessed_propensity_node",
                    ),
                    node(
                        func=read_climate.preprocess_data,
                        inputs=["raw_climate", "location"],
                        outputs="preprocessed_climate_data",
                        name="preprocessed_climate_node",
                    ),
                    # For merging
                    node(
                        func=merge_all_data,
                        inputs=[
                            "preprocessed_proxy_revenue_data",
                            "preprocessed_propensity_data",
                            "preprocessed_climate_data",
                            "preprocessed_covid_data",
                            "preprocessed_holiday_data",
                            "preprocessed_marketing_data",
                        ],
                        outputs="merged_df",  # output path at catalog.yml
                        name="merge_merge_all_data_node",
                    ),
                    node(
                        func=lambda merged_df, code=code,: save_data(merged_df, code),
                        inputs="merged_df",
                        outputs=None,
                        name="save_merged_df_node",
                    ),
                ]
            )
            # add cost code specific pipeline to auto_pipeline
            auto_pipeline += pipeline(
                data_loader,
                inputs={
                    "raw_propensity": "raw_propensity",
                    "raw_climate": "raw_climate",
                    "preprocessed_covid_data": "preprocessed_covid_data",
                    "preprocessed_holiday_data": "preprocessed_holiday_data",
                    "preprocessed_marketing_data": "preprocessed_marketing_data",
                },
                outputs=None,
                namespace=code,
            )
    return auto_pipeline
