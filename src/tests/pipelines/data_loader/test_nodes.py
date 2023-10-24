import pandas as pd
import numpy as np
import pytest
from bipo.utils import get_project_path
from kedro.config import ConfigLoader
from kedro.framework.project import settings

project_path = get_project_path()
# conf_loader = ConfigLoader(conf_source=project_path / "conf")
# conf_loader = ConfigLoader(conf_source=settings.PROJECT_PATH / settings.CONF_SOURCE)
# const_dict = conf_loader.get("constants*")

# DEFAULT CONSTANTS
# DEFAULT_DATE_COL = const_dict["default_date_col"]

# import dataloader node
from bipo.pipelines.data_loader.nodes import (
    load_and_partition_proxy_revenue_data,
    load_and_structure_marketing_data,
    load_and_structure_propensity_data,
    load_and_structure_weather_data,
    rename_merge_unique_csv_xlsx_df_col_index,
    merge_unique_csv_xlsx_df,
    rename_columns,
)

######### Add Fixtures Here ##################
# example dataset
# create or read a multi_daily_records or unique_daily_records file


# @pytest.fixture
# def empty_dataset():
#     return pd.DataFrame()  # empty dataframe


######### Test `rename_columns` ##############
def test_rename_columns():
    # Test with a string that contains spaces and non-alphanumeric characters
    input_string = " Date #$Q("
    expected_output = "date_q"
    assert rename_columns(input_string) == expected_output

    # Test with a string that contains leading and trailing spaces
    input_string = "  Leading and Trailing Spaces "  # will error if there is a "_" at the end of string
    expected_output = "leading_and_trailing_spaces"
    assert rename_columns(input_string) == expected_output

    # Test with an empty string (should return "")
    input_string = ""
    expected_output = ""
    assert rename_columns(input_string) == expected_output

    # Test with None (should return None)
    input_string = None
    expected_output = None
    assert rename_columns(input_string) == expected_output


######### Test `merge_df` assume outer join ##############
def test_merge_unique_csv_xlsx_df():  # data from unique_daily_records (holiday, covid data)
    sample_df_to_merge = pd.DataFrame(
        {
            "date": ["2021-01-01", "2021-01-02"],
            "Value1": [10, 20],
            "Value2": [30, 40],
        }
    )
    sample_df_to_merge_wrong_id = pd.DataFrame(
        {
            "wrong_id": ["2021-01-01", "2021-01-02"],
            "Value1": [10, 20],
            "Value2": [30, 40],
        }
    )
    sample_base_df = pd.DataFrame(
        {"date": ["2021-01-01", "2021-01-03"], "Value3": [50, 60]}
    )

    # Call the function to be tested
    combine_df = merge_unique_csv_xlsx_df(sample_df_to_merge, sample_base_df)

    # Check for expected columns in output result of the merge operation
    assert all(col in combine_df.columns for col in ["Value1", "Value2", "Value3"])

    # Check for expected shape in output result of the merge operation
    assert combine_df.shape == (3, 4)

    # Check output shape if expected index in not in df
    assert merge_unique_csv_xlsx_df(
        sample_df_to_merge_wrong_id, sample_base_df
    ).shape == (4, 5)


def test_rename_merge_unique_csv_xlsx_df_col_index():  # any data. This function is called after merge_df.
    sample_df = pd.DataFrame(
        {
            "Date": ["2021-01-01", "2021-01-02"],
            "Value1": [10, 20],
            "Value2": [30, 40],
        }
    )
    sample_df["Date"] = pd.to_datetime(sample_df["Date"])
    # Check when the date column is already in lowercase and present (if- statement)
    df1 = sample_df.copy()
    result_df1 = rename_merge_unique_csv_xlsx_df_col_index(df1)
    assert result_df1.index.name == "Date"

    # Check when the date column is missing, return None
    df2 = sample_df.copy()
    df2.drop(labels="Date", axis=1, inplace=True)
    result_df2 = rename_merge_unique_csv_xlsx_df_col_index(df2)
    assert result_df2.index.name == None

    # Check when neither the original nor renamed date-related column is present (else-statement)
    # df3 = sample_df.copy()
    # df3.drop(columns=["date"], inplace=True)
    # result_df3 = lowercase_set_date_on_partitions(df3)
    # assert result_df3.index.name is None


def test_load_and_partition_proxy_revenue_data():  # data from proxy reveune
    sample_proxy_reveune_df = pd.DataFrame(
        {
            "CostCentreCode": [201, 202],
            "Date": ["2022-01-01", "2022-01-02"],
            "ManHours": [31.2, 32.5],
            "ProxyRevenue": [1000.0, 2000.0],
            "Location": ["West", "East"],
            "Outlet": ["West-1", "East-1"],
            "Type": ["carry-out", "dine-in"],
        }
    )
    missing_costcentre_proxy_reveune_df = pd.DataFrame(
        {
            "MissingCostCentreCode": [201, 202],
            "Date": ["2022-01-01", "2022-01-02"],
            "ManHours": [31.2, 32.5],
            "ProxyRevenue": [1000.0, 2000.0],
            "Location": ["West", "East"],
            "Outlet": ["West-1", "East-1"],
            "Type": ["carry-out", "dine-in"],
        }
    )
    outlet_part_dict = load_and_partition_proxy_revenue_data(sample_proxy_reveune_df)
    # check if outlet_part_dict is a dict
    assert isinstance(outlet_part_dict, dict)

    # check if cost centre 201 and 202 is in the outlet dict
    # assert any(key in [201, 202] for key in outlet_part_dict.keys())
    assert "proxy_revenue_201" and "proxy_revenue_202" in outlet_part_dict

    # check if the number of columns is expected
    proxy_revenue_201_df = outlet_part_dict["proxy_revenue_201"]
    given_columns = proxy_revenue_201_df.columns
    assert len(given_columns) == 6

    # check if datatype of each column (numeric and string)
    # given_columns = outlet_part_dict["proxy_revenue_201"].columns
    assert proxy_revenue_201_df["costcentrecode"].dtype == int
    assert proxy_revenue_201_df["manhours"].dtype == float
    assert proxy_revenue_201_df["proxyrevenue"].dtype == float
    assert proxy_revenue_201_df["location"].dtype == object
    assert proxy_revenue_201_df["outlet"].dtype == object
    assert proxy_revenue_201_df["type"].dtype == object

    with pytest.raises(KeyError):
        load_and_partition_proxy_revenue_data(missing_costcentre_proxy_reveune_df)


def test_load_and_structure_propensity_data():  # data from propensity
    sample_propensity_df = pd.DataFrame(
        {
            "Date": [
                "2022-01-01",
                "2022-01-01",
                "2022-01-01",
                "2022-01-01",
                "2022-01-02",
            ],
            "Location": ["North", "South", "West", "East", "North"],
            "Factor": [1.0, 0, 1.75, 2.5, 1.2],
        }
    )
    # Generate function output
    groupby_df = load_and_structure_propensity_data(sample_propensity_df)

    # Check if Factor is mean value
    assert groupby_df.loc["2022-01-01", "factor"] == 1.3125

    # Check if the shape is as expected
    assert groupby_df.shape == (2, 1)  # 2 date row, 1 mean factor column

    # check if datatype of each column (numeric)
    assert groupby_df["factor"].dtype == float

    # Check if error is raise when "Factor" is missing
    with pytest.raises(KeyError):
        sample_propensity_df.drop(columns="Factor", inplace=True)
        load_and_structure_propensity_data(sample_propensity_df)


def test_load_and_structure_marketing_data():  # data from marketing
    # all sample df has a date range of 7 days
    sample_campaign1_df = pd.DataFrame(
        {
            "Name": ["campaign1"] * 7,
            "Date Start": ["2021-01-01"] * 7,
            "Date End": ["2021-01-07"] * 7,
            "Mode": [
                "TV Ad",
                "Radio Ad",
                "Instagram Ad",
                "Facebook Ad",
                "Youtube Ad",
                "Poster Campaign",
                "Digital",
            ],
            "Total Cost": [10000, 9000, np.nan, 7000, 6000, 5000, np.nan],
        }
    )
    # This df will be dropped in the function
    sample_campaign2_df = pd.DataFrame(
        {
            "Name": ["campaign2"] * 7,
            "Date Start": ["2021-01-01"] * 7,
            "Date End": ["2021-01-07"] * 7,
            "Mode": [
                "TV Ad",
                "Radio Ad",
                "Instagram Ad",
                "Facebook Ad",
                "Youtube Ad",
                "Poster Campaign",
                "Digital",
            ],
            "Total Cost": [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        }
    )
    # This df will be merged with df3 in the function
    sample_campaign3_df = pd.DataFrame(
        {
            "Name": ["campaign3"] * 7,
            "Date Start": ["2021-01-03"] * 7,
            "Date End": ["2021-01-07"] * 7,
            "Mode": [
                "TV Ad",
                "Radio Ad",
                "Instagram Ad",
                "Facebook Ad",
                "Youtube Ad",
                "Poster Campaign",
                "Digital",
            ],
            "Total Cost": [10000, 9000, 8000, 7000, 6000, 5000, 4000],
        }
    )
    # merge all campaign
    marketing_df = pd.concat(
        [sample_campaign1_df, sample_campaign2_df, sample_campaign3_df],
        axis=0,
        join="outer",
    )

    marketing_df["Date End"] = pd.to_datetime(marketing_df["Date End"])
    marketing_df["Date Start"] = pd.to_datetime(marketing_df["Date Start"])

    df = load_and_structure_marketing_data(marketing_df)

    # Check if there is 7 days
    assert len(df.index) == 7

    # Check if there is no NaN values
    assert df.isna().any().any() == False

    # Check if there are more than 1 campaign, "name" will have a list with 2 values
    assert len(df.loc["2021-01-03", "name"]) == 2

    # Check if there are more than 1 campaign, "facebook_ad" will be sum of 2 campaign daily cost 2400.0
    assert df.loc["2021-01-03", "facebook_ad_daily_cost"] == 2400.0

    # Check if campaign 2 is removed. Not in "name"
    for i in df.index:
        if "campaign2" not in df.loc[i, "name"]:
            assert True  # Assert that the condition is True

    # Check final output df is datatype (numeric) for daily cost columns
    assert df["name"].dtype == list
    assert df["tv_ad_daily_cost"].dtype == float
    assert df["radio_ad_daily_cost"].dtype == float
    assert df["instagram_ad_daily_cost"].dtype == float
    assert df["facebook_ad_daily_cost"].dtype == float
    assert df["youtube_ad_daily_cost"].dtype == float
    assert df["poster_campaign_daily_cost"].dtype == float
    assert df["digital_daily_cost"].dtype == float

    # Check if error is raise when mkt columns are missing
    with pytest.raises(KeyError):
        marketing_df.drop(columns="Date Start", inplace=True)
        load_and_structure_marketing_data(marketing_df)


def test_load_and_structure_weather_data():  # data from weather
    # all sample df has a date range of 7 days
    sample_weather_df = pd.DataFrame(
        {
            "DIRECTION": ["North", "South", "East", "West", "North"],
            "Station": [
                "Admiralty",
                "Marina Barrage",
                "Changi",
                "Jurong (West)",
                "Admiralty",
            ],
            "Year": [2021, 2021, 2021, 2021, 2022],
            "Month": [1, 1, 1, 1, 1],
            "Day": [1, 1, 1, 1, 1],
            "Daily Rainfall Total (mm)": [10.2, 5.5, 15.0, 7.8, 12.3],
            "Highest 30 min Rainfall (mm)": [3.2, 2.1, 5.5, 1.8, 4.6],
            "Highest 60 min Rainfall (mm)": [5.7, 4.3, 7.8, 3.5, 6.2],
            "Highest 120 min Rainfall (mm)": [8.9, 7.2, 10.3, 5.6, 9.4],
            "Mean Temperature (°C)": [28.1, 29.5, 26.8, 27.3, 30.0],
            "Maximum Temperature (°C)": [32.4, 33.8, 30.5, 31.2, 34.5],
            "Minimum Temperature (°C)": [24.2, 25.6, 23.6, 24.8, 26.7],
            "Mean Wind Speed (km/h)": [12.1, 10.5, 14.3, 11.8, 13.2],
            "Max Wind Speed (km/h)": [18.7, 17.2, 20.4, 19.5, 21.0],
        }
    )
    # Check if the output is mean
    df = load_and_structure_weather_data(sample_weather_df)
    assert df.loc["2021-01-01", "highest_120_min_rainfall_mm"] == 8.0

    # Check final output df is datatype (numeric) for weather columns
    # these 2 columns contains NaN
    # assert df["direction"].dtype == float
    # assert df["station"].dtype == float
    assert df["daily_rainfall_total_mm"].dtype == float
    assert df["highest_30_min_rainfall_mm"].dtype == float
    assert df["highest_120_min_rainfall_mm"].dtype == float
    assert df["mean_temperature_c"].dtype == float
    assert df["maximum_temperature_c"].dtype == float
    assert df["minimum_temperature_c"].dtype == float
    assert df["mean_wind_speed_kmh"].dtype == float
    assert df["max_wind_speed_kmh"].dtype == float

    # Check if error is raise when mkt columns are missing
    with pytest.raises(KeyError):
        sample_weather_df.drop(columns="Year", inplace=True)
        load_and_structure_weather_data(sample_weather_df)
