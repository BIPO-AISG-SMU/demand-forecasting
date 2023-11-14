from typing import List, Dict
from bipo.pipelines.data_preprocessing.nodes import (
    outlet_exclusion_list_check,
    const_value_perc_check,
    date_validity_check,
    merge_non_proxy_revenue_data,
    merge_outlet_and_other_df_feature,
)

import pytest
import pandas as pd
from pandas.testing import assert_frame_equal

# Global variable for parameters dictionary
PARAMS_DICT = {
    "start_date": "2021-01-01",
    "end_date": "2022-12-31",
    "zero_val_threshold_perc": 2,
    "outlets_exclusion_list": [],
}


class TestDataPreprocessingNode:
    # Factory method to generate dataset
    @pytest.fixture(scope="module")
    def create_preprocessing_datapoints(self):
        def _create_preprocessing_datapoints(
            datapoint_rows: Dict, index=None, columns=None
        ):
            return pd.DataFrame(datapoint_rows, index=index, columns=columns)

        return _create_preprocessing_datapoints

    @pytest.fixture
    def default_values(self):
        """Fixture to provide default configuration values."""
        config = {
            "default_start_date": "2021-01-01",
            "default_end_date": "2022-12-31",
            "default_revenue_column": "proxyrevenue",
            "default_const_value_perc_threshold": 0,
            "default_outlets_exclusion_list": [],
        }
        return config

    ######### Test outlet_exclusion_list_check ##############
    def test_outlet_exclusion_list_check_correct_input(self):
        # Test for valid outlet exclusion list
        assert outlet_exclusion_list_check([1, 2, 3]) == {"1", "2", "3"}

    def test_outlet_exclusion_list_check_incorrect_input(self, default_values):
        # Test for invalid outlet exclusion list
        assert outlet_exclusion_list_check("string") == set(
            default_values["default_outlets_exclusion_list"]
        )

    def test_outlet_exclusion_list_check_correct_output_type(self):
        # Test for correct output type of outlet exclusion list check
        output = outlet_exclusion_list_check([1, 2, 3])
        assert isinstance(output, set)

    ######### Test const_value_perc_check ##############
    def test_const_value_perc_check_within_range(self):
        # Test for constant value percentage within valid range
        assert const_value_perc_check(25) == 25.0

    def test_const_value_perc_check_outside_range(self):
        # Test for constant value percentage outside valid range
        assert const_value_perc_check(105) == 100.0
        assert const_value_perc_check(-2) == 0.0

    def test_const_value_perc_check_incorrect_input(self, default_values):
        # Test for invalid input in constant value percentage check
        assert (
            const_value_perc_check("string")
            == default_values["default_const_value_perc_threshold"]
        )

    ######### Test date_validity_check ##############
    def test_date_validity_check_correct_dates(self):
        # Test for date validity with correct date format
        assert date_validity_check("2021-01-01", "2021-12-31") == (
            "2021-01-01",
            "2021-12-31",
        )

    def test_date_validity_check_inverted_dates(self):
        # Test for date validity with inverted dates
        assert date_validity_check("2021-12-31", "2021-01-01") == (
            "2021-01-01",
            "2021-12-31",
        )

    def test_date_validity_check_incorrect_format(self, default_values):
        # Test for invalid date format in date validity check
        assert date_validity_check("2021-31-31", "2021-01-01") == (
            default_values["default_start_date"],
            default_values["default_end_date"],
        )

    ######### Test merge_non_proxy_revenue_data ##############
    @pytest.fixture
    def mock_non_revenue(self, create_preprocessing_datapoints):
        # Fixture to mock non-revenue data
        df = create_preprocessing_datapoints(
            {
                "Date": ["2021-01-01", "2021-01-02", "2021-01-03"],
                "factor": [1.75, 1.5, 1.0],
            }
        )
        df.set_index(pd.to_datetime(df["Date"]), inplace=True)
        return df

    @pytest.fixture
    def mock_non_revenue_diff_columns(self, create_preprocessing_datapoints):
        # Fixture to mock non-revenue data with different columns
        df = create_preprocessing_datapoints(
            {
                "Date": ["2021-01-01", "2021-01-02", "2021-01-03"],
                "group_size_cap": [8, 5, 2],
                "school_holiday": ["Yes", "No", "Yes"],
            }
        )
        df.set_index(pd.to_datetime(df["Date"]), inplace=True)
        return df

    @pytest.fixture
    def mock_non_revenue_diff_dates(self, create_preprocessing_datapoints):
        # Fixture to mock non-revenue data with differing dates
        df = create_preprocessing_datapoints(
            {
                "Date": ["2021-12-01", "2021-12-02", "2021-12-03"],
                "factor": [1.75, 1.5, 1.0],
            }
        )
        df.set_index(pd.to_datetime(df["Date"]), inplace=True)
        return df

    def test_merge_non_proxy_revenue_data_single_dataframe(
        self, mock_non_revenue_diff_dates
    ):
        # Test merging non-proxy revenue data from a single DataFrame
        partitioned_input = {"partition_1": mock_non_revenue_diff_dates}
        merged_df = merge_non_proxy_revenue_data(partitioned_input)
        assert_frame_equal(merged_df, mock_non_revenue_diff_dates)

    def test_merge_non_proxy_revenue_data_multiple_dataframes(
        self, mock_non_revenue, mock_non_revenue_diff_columns
    ):
        # Test merging non-proxy revenue data from multiple DataFrames
        partitioned_input = {
            "mock_non_revenue": mock_non_revenue,
            "mock_non_revenue_diff_columns": mock_non_revenue_diff_columns,
        }
        merged_df = merge_non_proxy_revenue_data(partitioned_input)

        expected_df = pd.concat(
            [mock_non_revenue_diff_columns, mock_non_revenue],
            axis=1,
            join="outer",
            ignore_index=False,
        )
        assert_frame_equal(merged_df, expected_df)

    def test_merge_non_proxy_revenue_data_multiple_dataframes_diff_dates(
        self, mock_non_revenue_diff_columns, mock_non_revenue_diff_dates
    ):
        # Test merging non-proxy revenue data from DataFrames with differing dates
        partitioned_input = {
            "mock_non_revenue_diff_columns": mock_non_revenue_diff_columns,
            "mock_non_revenue_diff_dates": mock_non_revenue_diff_dates,
        }
        merged_df = merge_non_proxy_revenue_data(partitioned_input)

        expected_df = pd.concat(
            [mock_non_revenue_diff_dates, mock_non_revenue_diff_columns],
            axis=1,
            join="outer",
            ignore_index=False,
        )
        assert_frame_equal(merged_df, expected_df)

    ######### Test merge_outlet_and_other_df_feature ##############
    @pytest.fixture
    def mock_revenue_partitioned(self, create_preprocessing_datapoints):
        # Fixture to mock partitioned revenue data
        df = create_preprocessing_datapoints(
            {
                "Date": ["2021-01-01", "2021-01-02", "2021-01-03"],
                "proxyrevenue": [4520.0, 3530.0, 4250.0],
            }
        )
        df.set_index(df["Date"], inplace=True)

        outlet_partitioned = {"outlet_1": df}
        return outlet_partitioned

    def test_merge_outlet_and_other_df_feature_correct_output_type(
        self, mock_revenue_partitioned, mock_non_revenue
    ):
        # Test for verifying the output type after merging outlet and other data
        result = merge_outlet_and_other_df_feature(
            mock_revenue_partitioned, mock_non_revenue, PARAMS_DICT
        )
        assert isinstance(result, dict)

    def test_merge_outlet_and_other_df_feature_zero_value_threshold(
        self, mock_revenue_partitioned, mock_non_revenue
    ):
        # Test for checking exclusion based on zero value threshold
        mock_revenue_partitioned["outlet_1"]["proxyrevenue"] = 0
        result = merge_outlet_and_other_df_feature(
            mock_revenue_partitioned, mock_non_revenue, PARAMS_DICT
        )
        assert "outlet_1_processed" not in result.keys()

    def test_merge_outlet_and_other_df_feature_empty_revenue_partitioned(
        self,
        mock_non_revenue,
    ):
        # Test for handling empty partitioned revenue data
        result = merge_outlet_and_other_df_feature({}, mock_non_revenue, PARAMS_DICT)
        assert result == {}
