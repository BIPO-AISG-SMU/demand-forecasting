from typing import List, Dict
import pandas as pd
from pandas.testing import assert_frame_equal
import pytest

from bipo.pipelines.time_agnostic_feature_engineering.feature_indicator_diff_creation import (
    create_min_max_feature_diff,
    create_is_weekday_feature,
    create_is_holiday_feature,
    create_is_raining_feature,
    create_is_pandemic_feature,
    create_marketing_counts_start_end_features,
)

class TestTimeAgnosticFeatureEngineeringNode:

    # Factory method to generate dataset
    @pytest.fixture(scope="module")
    def create_preprocessing_datapoints(self):
        def _create_preprocessing_datapoints(
            datapoint_rows: List[Dict], index=None, columns=None
        ):
            return pd.DataFrame(datapoint_rows, index=index, columns=columns)

        return _create_preprocessing_datapoints

    def test_create_min_max_feature_diff(self, create_preprocessing_datapoints):
        data_points_df = create_preprocessing_datapoints(
            [
                {
                    "minimum_temperature_c": 26.5,
                    "maximum_temperature_c": 29,
                    "mean_wind_speed_kmh": 1,
                    "max_wind_speed_kmh": 2,
                },
                {
                    "minimum_temperature_c": 29.5,
                    "maximum_temperature_c": 31.2,
                    "mean_wind_speed_kmh": 2,
                    "max_wind_speed_kmh": 4,
                },
                {
                    "minimum_temperature_c": 30.3,
                    "maximum_temperature_c": 30.6,
                    "mean_wind_speed_kmh": 0,
                    "max_wind_speed_kmh": 0,
                },
            ],
            index=pd.to_datetime(
                ["2020-12-31", "2021-01-01", "2021-01-02"], format="%Y-%m-%d"
            ),
            columns=[
                "minimum_temperature_c",
                "maximum_temperature_c",
                "mean_wind_speed_kmh",
                "max_wind_speed_kmh",
            ],
        )

        data_points_df = create_min_max_feature_diff(
            df=data_points_df,
            min_max_column_list=[
                ["minimum_temperature_c", "maximum_temperature_c"],
                ["mean_wind_speed_kmh", "max_wind_speed_kmh"],
            ],
        )

        expected_data_points_df = create_preprocessing_datapoints(
            [
                {
                    "minimum_temperature_c": 26.5,
                    "maximum_temperature_c": 29,
                    "mean_wind_speed_kmh": 1,
                    "max_wind_speed_kmh": 2,
                    "diff_minimum_temperature_c_maximum_temperature_c": 2.5,
                    "diff_mean_wind_speed_kmh_max_wind_speed_kmh": 1,
                },
                {
                    "minimum_temperature_c": 29.5,
                    "maximum_temperature_c": 31.2,
                    "mean_wind_speed_kmh": 2,
                    "max_wind_speed_kmh": 4,
                    "diff_minimum_temperature_c_maximum_temperature_c": 1.7,
                    "diff_mean_wind_speed_kmh_max_wind_speed_kmh": 2,
                },
                {
                    "minimum_temperature_c": 30.3,
                    "maximum_temperature_c": 30.6,
                    "mean_wind_speed_kmh": 0,
                    "max_wind_speed_kmh": 0,
                    "diff_minimum_temperature_c_maximum_temperature_c": 0.3,
                    "diff_mean_wind_speed_kmh_max_wind_speed_kmh": 0,
                },
            ],
            index=pd.to_datetime(
                ["2020-12-31", "2021-01-01", "2021-01-02"], format="%Y-%m-%d"
            ),
            columns=[
                "minimum_temperature_c",
                "maximum_temperature_c",
                "mean_wind_speed_kmh",
                "max_wind_speed_kmh",
                "diff_minimum_temperature_c_maximum_temperature_c",
                "diff_mean_wind_speed_kmh_max_wind_speed_kmh",
            ],
        )

        assert_frame_equal(data_points_df, expected_data_points_df, check_like=True)

    def test_create_is_weekday_feature(self, create_preprocessing_datapoints):
        data_points_df = create_preprocessing_datapoints(
            [],
            index=pd.to_datetime(
                ["2020-12-31", "2021-01-01", "2021-01-02"], format="%Y-%m-%d"
            ),
        )

        data_points_df = create_is_weekday_feature(data_points_df)

        expected_data_points_df = create_preprocessing_datapoints(
            [
                {"is_weekday": 1},
                {"is_weekday": 1},
                {"is_weekday": 0},
            ],
            index=pd.to_datetime(
                ["2020-12-31", "2021-01-01", "2021-01-02"], format="%Y-%m-%d"
            ),
            columns=["is_weekday"],
        )

        assert_frame_equal(data_points_df, expected_data_points_df, check_like=True)

    def test_create_is_holiday_feature(self, create_preprocessing_datapoints):
        data_points_df = create_preprocessing_datapoints(
            [
                {
                    "school_holiday": "Yes",
                    "public_holiday": None,
                },
                {
                    "school_holiday": None,
                    "public_holiday": "Yes",
                },
                {
                    "school_holiday": None,
                    "public_holiday": None,
                },
            ],
            index=pd.to_datetime(
                ["2020-12-31", "2021-01-01", "2021-01-02"], format="%Y-%m-%d"
            ),
            columns=["school_holiday", "public_holiday"],
        )

        data_points_df = create_is_holiday_feature(
            df=data_points_df,
            holiday_type_col_list=["school_holiday", "public_holiday"],
        )

        expected_data_points_df = create_preprocessing_datapoints(
            [
                {
                    "school_holiday": "Yes",
                    "public_holiday": None,
                    "is_school_holiday": 1,
                    "is_public_holiday": 0,
                },
                {
                    "school_holiday": None,
                    "public_holiday": "Yes",
                    "is_school_holiday": 0,
                    "is_public_holiday": 1,
                },
                {
                    "school_holiday": None,
                    "public_holiday": None,
                    "is_school_holiday": 0,
                    "is_public_holiday": 0,
                },
            ],
            index=pd.to_datetime(
                ["2020-12-31", "2021-01-01", "2021-01-02"], format="%Y-%m-%d"
            ),
            columns=[
                "school_holiday",
                "public_holiday",
                "is_school_holiday",
                "is_public_holiday",
            ],
        )

        assert_frame_equal(data_points_df, expected_data_points_df, check_like=True)

    def test_create_is_raining_feature(self, create_preprocessing_datapoints):
        # Threshold below and above 0.2
        data_points_df = create_preprocessing_datapoints(
            [
                {"daily_rainfall_total_mm": 15},
                {"daily_rainfall_total_mm": 0},
                {"daily_rainfall_total_mm": 0.2},
            ],
            index=pd.to_datetime(
                ["2020-12-31", "2021-01-01", "2021-01-02"], format="%Y-%m-%d"
            ),
            columns=["daily_rainfall_total_mm"],
        )

        data_points_df = create_is_raining_feature(
            df=data_points_df, rainfall_col="daily_rainfall_total_mm"
        )

        expected_data_points_df = create_preprocessing_datapoints(
            [
                {
                    "daily_rainfall_total_mm": 15,
                    "is_daily_rainfall_total_mm": 1,
                },
                {
                    "daily_rainfall_total_mm": 0,
                    "is_daily_rainfall_total_mm": 0,
                },
                {
                    "daily_rainfall_total_mm": 0.2,
                    "is_daily_rainfall_total_mm": 0,
                },
            ],
            index=pd.to_datetime(
                ["2020-12-31", "2021-01-01", "2021-01-02"], format="%Y-%m-%d"
            ),
            columns=["daily_rainfall_total_mm", "is_daily_rainfall_total_mm"],
        )

        assert_frame_equal(data_points_df, expected_data_points_df, check_like=True)

    def test_create_is_pandemic_feature(self, create_preprocessing_datapoints):
        # Test diff combinations of group_size
        data_points_df = create_preprocessing_datapoints(
            [
                {"group_size": "2"},
                {"group_size": "5"},
                {"group_size": "no limit"},
            ],
            index=pd.to_datetime(
                ["2020-12-31", "2021-01-01", "2021-01-02"], format="%Y-%m-%d"
            ),
            columns=["group_size"],
        )

        data_points_df = create_is_pandemic_feature(
            df=data_points_df, pandemic_col="group_size"
        )
        expected_data_points_df = create_preprocessing_datapoints(
            [
                {
                    "group_size": "2",
                    "is_pandemic_restrictions": 1,
                },
                {
                    "group_size": "5",
                    "is_pandemic_restrictions": 1,
                },
                {
                    "group_size": "no limit",
                    "is_pandemic_restrictions": 0,
                },
            ],
            index=pd.to_datetime(
                ["2020-12-31", "2021-01-01", "2021-01-02"], format="%Y-%m-%d"
            ),
            columns=["group_size", "is_pandemic_restrictions"],
        )

        assert_frame_equal(data_points_df, expected_data_points_df, check_like=True)

    def test_create_marketing_start_end_features(self, create_preprocessing_datapoints):
        # Test diff of event occurrence (start/stop)
        data_points_df = create_preprocessing_datapoints(
            [
                {"name": "['event0']"},
                {"name": "['event1']"},
                {"name": ""},
                {"name": "['event2']"},
                {"name": "['event2','event3']"},
                {"name": "['event3']"},
            ],
            index=pd.to_datetime(
                [
                    "2020-12-29",
                    "2020-12-30",
                    "2020-12-31",
                    "2021-01-01",
                    "2021-01-02",
                    "2021-01-03",
                ],
                format="%Y-%m-%d",
            ),
            columns=["name"],
        )

        data_points_df = create_marketing_counts_start_end_features(
            df=data_points_df,
            mkt_campaign_col="name",
            mkt_count_col_name="name_counts",
            mkt_start="is_name_start",
            mkt_end="is_name_end",
        )

        expected_data_points_df = create_preprocessing_datapoints(
            [
                {
                    "name_counts": 1,
                    "is_name_start": 0,
                    "is_name_end": 1,
                },
                {
                    "name_counts": 1,
                    "is_name_start": 1,
                    "is_name_end": 1,
                },
                {
                    "name_counts": 0,
                    "is_name_start": 0,
                    "is_name_end": 0,
                },
                {
                    "name_counts": 1,
                    "is_name_start": 1,
                    "is_name_end": 0,
                },
                {
                    "name_counts": 2,
                    "is_name_start": 1,
                    "is_name_end": 1,
                },
                {
                    "name_counts": 1,
                    "is_name_start": 0,
                    "is_name_end": 0,
                },
            ],
            index=pd.to_datetime(
                [
                    "2020-12-29",
                    "2020-12-30",
                    "2020-12-31",
                    "2021-01-01",
                    "2021-01-02",
                    "2021-01-03",
                ],
                format="%Y-%m-%d",
            ),
            columns=["name_counts", "is_name_start", "is_name_end"],
        )

        assert_frame_equal(data_points_df, expected_data_points_df, check_like=True)
