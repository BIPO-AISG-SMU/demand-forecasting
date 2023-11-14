from typing import List, Dict
import pandas as pd
from pandas.testing import assert_frame_equal
import pytest


from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.preprocessing import Normalizer, StandardScaler

## Encodings
from bipo.pipelines.feature_engineering.encoding import (
    ordinal_encoding_transform,
    one_hot_encoding_transform,
)

## Lag generations
from bipo.pipelines.feature_engineering.lag_feature_generation import (
    create_simple_lags,
    create_sma_lags,
    # create_lag_weekly_avg_sales,
)

## Normalization
from bipo.pipelines.feature_engineering.standardize_normalize import (
    standard_norm_transform,
)


class TestFeatureEngineeringNode:
    @pytest.fixture(scope="module")
    def create_preprocessing_datapoints(self):
        def _create_preprocessing_datapoints(
            datapoint_rows: List[Dict], index=None, columns=None
        ):
            return pd.DataFrame(datapoint_rows, index=index, columns=columns)

        return _create_preprocessing_datapoints

    def test_ordinal_encoding_transform(self, create_preprocessing_datapoints):
        data_points_df = create_preprocessing_datapoints(
            [
                {"type": "dine-in"},
                {"type": "carry-out"},
                {"type": "dine-in"},
                {"type": "carry-out"},
            ],
            index=pd.to_datetime(
                [
                    "2020-12-30",
                    "2020-12-31",
                    "2021-01-01",
                    "2021-01-02",
                ],
                format="%Y-%m-%d",
            ),
            columns=[
                "type",
            ],
        )

        ord_encoder = OrdinalEncoder(
            categories=[
                ["carry-out", "dine-in"],
            ],
            handle_unknown="use_encoded_value",
            unknown_value=-1,  # For unknown categories
            encoded_missing_value=-1,  # For missing values
        )

        ord_encoder.fit(data_points_df[["type"]])

        data_points_df = ordinal_encoding_transform(
            df=data_points_df, ordinal_encoder=ord_encoder
        )

        expected_data_points_df = create_preprocessing_datapoints(
            [
                {"ord_type": 1.0, "type": "dine-in"},
                {"ord_type": 0.0, "type": "carry-out"},
                {"ord_type": 1.0, "type": "dine-in"},
                {"ord_type": 0.0, "type": "carry-out"},
            ],
            index=pd.to_datetime(
                [
                    "2020-12-30",
                    "2020-12-31",
                    "2021-01-01",
                    "2021-01-02",
                ],
                format="%Y-%m-%d",
            ),
            columns=[
                "ord_type",
                "type",
            ],
        )

        assert_frame_equal(data_points_df, expected_data_points_df, check_like=True)

    def test_one_hot_encoding_transform(self, create_preprocessing_datapoints):
        data_points_df = create_preprocessing_datapoints(
            [
                {"type": "dine-in"},
                {"type": "carry-out"},
                {"type": "dine-in"},
                {"type": "carry-out"},
            ],
            index=pd.to_datetime(
                [
                    "2020-12-30",
                    "2020-12-31",
                    "2021-01-01",
                    "2021-01-02",
                ],
                format="%Y-%m-%d",
            ),
            columns=[
                "type",
            ],
        )

        # Simulate the use of one_hot_encoding_fit, where first category(alphanumeric ordering) is dropped.
        ohe_encoder = OneHotEncoder(
            categories="auto",
            handle_unknown="ignore",
            drop="first",
            sparse_output=False,
        ).fit(data_points_df[["type"]])

        data_points_df = one_hot_encoding_transform(
            df=data_points_df, ohe_encoder=ohe_encoder
        )
        print(data_points_df)
        # For one-hot encoding, feature name is constructed with <col_name>_<value>
        expected_data_points_df = create_preprocessing_datapoints(
            [
                {"type_dine-in": 1.0},
                {"type_dine-in": 0.0},
                {"type_dine-in": 1.0},
                {"type_dine-in": 0.0},
            ],
            index=pd.to_datetime(
                [
                    "2020-12-30",
                    "2020-12-31",
                    "2021-01-01",
                    "2021-01-02",
                ],
                format="%Y-%m-%d",
            ),
            columns=[
                "type_dine-in",
            ],
        )

        assert_frame_equal(data_points_df, expected_data_points_df, check_like=True)

    def test_create_simple_lags(self, create_preprocessing_datapoints):
        data_points_df = create_preprocessing_datapoints(
            [
                {"proxyrevenue": 10.5},
                {"proxyrevenue": 9.5},
                {"proxyrevenue": 8.5},
                {"proxyrevenue": 7.5},
                {"proxyrevenue": 6.5},
                {"proxyrevenue": 5.5},
            ],
            index=pd.to_datetime(
                [
                    "2021-12-30",
                    "2020-12-31",
                    "2021-01-01",
                    "2021-01-02",
                    "2021-01-03",
                    "2021-01-04",
                ],
                format="%Y-%m-%d",
            ),
            columns=[
                "proxyrevenue",
            ],
        )

        data_points_df = create_simple_lags(df=data_points_df, lag_periods_list=[1, 2])
        # Create expected output using lag 2
        expected_data_points_df = create_preprocessing_datapoints(
            [
                {
                    "lag_1_proxyrevenue": 9.5,
                    "lag_2_proxyrevenue": 10.5,
                },
                {
                    "lag_1_proxyrevenue": 8.5,
                    "lag_2_proxyrevenue": 9.5,
                },
                {
                    "lag_1_proxyrevenue": 7.5,
                    "lag_2_proxyrevenue": 8.5,
                },
                {
                    "lag_1_proxyrevenue": 6.5,
                    "lag_2_proxyrevenue": 7.5,
                },
            ],
            index=pd.to_datetime(
                [
                    "2021-01-01",
                    "2021-01-02",
                    "2021-01-03",
                    "2021-01-04",
                ],
                format="%Y-%m-%d",
            ),
            columns=["lag_1_proxyrevenue", "lag_2_proxyrevenue"],
        )

        assert_frame_equal(data_points_df, expected_data_points_df, check_like=True)

    def test_create_sma_lags(self, create_preprocessing_datapoints):
        data_points_df = create_preprocessing_datapoints(
            [
                {"proxyrevenue": 10},
                {"proxyrevenue": 9},
                {"proxyrevenue": 8},
                {"proxyrevenue": 7},
                {"proxyrevenue": 6},
                {"proxyrevenue": 5},
            ],
            index=pd.to_datetime(
                [
                    "2020-12-30",
                    "2020-12-31",
                    "2021-01-01",
                    "2021-01-02",
                    "2021-01-03",
                    "2021-01-04",
                ],
                format="%Y-%m-%d",
            ),
            columns=[
                "proxyrevenue",
            ],
        )

        data_points_df = create_sma_lags(
            df=data_points_df, shift_period=1, sma_window_period_list=[2]
        )
        # Create expected output using lag 2
        expected_data_points_df = create_preprocessing_datapoints(
            [
                {
                    "lag_1_sma_2_days_proxyrevenue": 9.5,
                },
                {
                    "lag_1_sma_2_days_proxyrevenue": 8.5,
                },
                {
                    "lag_1_sma_2_days_proxyrevenue": 7.5,
                },
                {
                    "lag_1_sma_2_days_proxyrevenue": 6.5,
                },
            ],
            index=pd.to_datetime(
                [
                    "2021-01-01",
                    "2021-01-02",
                    "2021-01-03",
                    "2021-01-04",
                ],
                format="%Y-%m-%d",
            ),
            columns=[
                "lag_1_sma_2_days_proxyrevenue",
            ],
        )

        assert_frame_equal(data_points_df, expected_data_points_df, check_like=True)

    # Tests for standardscaler
    def test_standard_transform(self, create_preprocessing_datapoints):
        data_points_df = create_preprocessing_datapoints(
            [
                {"proxyrevenue": 0},
                {"proxyrevenue": 0},
                {"proxyrevenue": 1},
                {"proxyrevenue": 1},
            ],
            index=pd.to_datetime(
                ["2020-12-30", "2020-12-31", "2021-01-01", "2021-01-02"],
                format="%Y-%m-%d",
            ),
            columns=[
                "proxyrevenue",
            ],
        )
        std_scaler = StandardScaler()
        # Apply fit based on instantiated normalizer
        std_scaler_fit = std_scaler.fit(data_points_df[["proxyrevenue"]])

        data_points_df = standard_norm_transform(
            df=data_points_df, std_norm_object=std_scaler_fit
        )
        # Create expected output by moc
        expected_data_points_df = create_preprocessing_datapoints(
            [
                {"proxyrevenue": -1},
                {"proxyrevenue": -1},
                {"proxyrevenue": 1},
                {"proxyrevenue": 1},
            ],
            index=pd.to_datetime(
                ["2020-12-30", "2020-12-31", "2021-01-01", "2021-01-02"],
                format="%Y-%m-%d",
            ),
            columns=[
                "proxyrevenue",
            ],
        )

        assert_frame_equal(data_points_df, expected_data_points_df, check_like=True)

    # Tests for normalizer
    def test_normalizer_transform(self, create_preprocessing_datapoints):
        data_points_df = create_preprocessing_datapoints(
            [
                {"proxyrevenue": 4},
                {"proxyrevenue": 1},
                {"proxyrevenue": 2},
                {"proxyrevenue": 2},
            ],
            index=pd.to_datetime(
                ["2020-12-30", "2020-12-31", "2021-01-01", "2021-01-02"],
                format="%Y-%m-%d",
            ),
            columns=[
                "proxyrevenue",
            ],
        )
        norm_scaler = Normalizer()
        # Apply fit based on instantiated normalizer
        norm_scaler_fit = norm_scaler.fit(data_points_df[["proxyrevenue"]])

        data_points_df = standard_norm_transform(
            df=data_points_df, std_norm_object=norm_scaler_fit
        )
        # Create expected output which normalizer treats each row as separate groups.
        expected_data_points_df = create_preprocessing_datapoints(
            [
                {"proxyrevenue": 1},
                {"proxyrevenue": 1},
                {"proxyrevenue": 1},
                {"proxyrevenue": 1},
            ],
            index=pd.to_datetime(
                ["2020-12-30", "2020-12-31", "2021-01-01", "2021-01-02"],
                format="%Y-%m-%d",
            ),
            columns=[
                "proxyrevenue",
            ],
        )

        assert_frame_equal(data_points_df, expected_data_points_df, check_like=True)
