# Data Pipeline Configuration

## Overview

This section contains information on key parameters from the files `parameters.yml`, `constants.yml` from `conf/base/` directory as well as yml files in `conf/base/parameters/` subdirectory, namely  `data_split.yml` file. Due to the volume of parameters, we will only include a selection for illustrative purposes. These files can be found in `conf/base/` directory.

### Location of Configuration Files

There are two separate directories containing .yml files, namely in `conf/base/` and `conf/base/parameters`. Those placed in `conf/base` are primarily shared across different Kedro pipelines, to facilitate single configuration entrypoint that can be used across the entire pipeline. On the other hand, files located in `conf/base/parameters` are identified by its module name naming convention (e.g. `data_split.yml`) and govern the configuration for a specific pipeline, so as to facilitate experimentation of configuration values.

## Key Parameters in `parameters.yml`

This sub-section outlines the key parameters in the `parameters.yml` configuration file that control different aspects of the data pipeline. The file is divided into several sections, each dealing with a specific part of the pipeline. 

### Data Loader

- Not applicable

### Data Preprocessing

| Parameter | Type | Description | Default Value |
| --- | --- | --- | --- |
| `start_date` | `str` | Data start in `yyyy-mm-dd` format. | '2021-01-01' |
| `end_date` | `str` | Data end in `yyyy-mm-dd` format. | '2022-12-31' |
| `zero_val_threshold_perc` | `float`  | Percentage of zeroed-values allowed (0 to 100 only). | 2 |
| `outlets_exclusion_list` | `list` | List of outlets (costcentercode) represented in strings to be excluded. Ideally should be the values found in data. <br /> Example: ['201', '202'] | [ ] |
---

### Data Merge Configurations

- Not applicable

### Time Agnostic Feature Engineering

| Parameter | Type | Description | Default Value |
| --- | --- | --- | --- |
| `fe_columns_to_drop_list` | `list` | Columns to be dropped after time-agnostic feature engineering is completed. It is advised to refer to the data artefacts generated after data merge module as it would indicate the columns available to be considered for necessary feature engineering (except index itself). | ['day', 'location', 'manhours', 'outlet', 'highest_30_min_rainfall_mm', 'highest_60_min_rainfall_mm', 'highest_120_min_rainfall_mm', 'mean_temperature_c', 'min_temperature_c', 'mean_wind_speed_kmh', 'max_wind_speed_kmh', 'school_holiday_type', 'public_holiday_type'] |
| `fe_mkt_column_name` | `str` | Column name for marketing-related features. | 'name' |
| `mkt_columns_to_impute_dict` | `dict` | Dictionary containing marketing features representing daily cost and values used for imputation. This is targeted towards dates where there are no marketing events. | {'tv_ad_daily_cost': 0, 'radio_ad_daily_cost': 0, 'instagram_ad_daily_cost': 0, 'facebook_ad_daily_cost': 0, 'youtube_ad_daily_cost': 0, 'poster_campaign_daily_cost': 0, 'digital_daily_cost': 0} |
| `split_approach_source` | `str` | Source of data split to be used for feature engineering processes and beyond. Either `simple_split`, `expanding_window` or `sliding_window`. | 'simple_split' |
| `fe_rainfall_column` | `str` | Column representing daily total rainfall for the purpose of generating boolean indicator `is_raining` feature. <br /><br />**Condition is hardcoded based on assuming values exceeding 0.2 indicates a rainy day.** | 'daily_rainfall_total_mm'
| `fe_holiday_column_list` | `list` | Columns referencing school and public holiday columns for the purpose of generating boolean indicator feature prefixed with `is_`. <br /><br /> **Condition is hardcoded based on assuming entries of the column as indication of non-holiday, Otherwise it indicates holiday.** | ['school_holiday', 'public_holiday'] |
| `fe_pandemic_column` | `str` | Column representing pandemic restrictions imposed in terms of group size amount for generating a boolean indicator feature `is_pandemic_restrictions`. <br /><br />**Condition is hardcoded by checking if values = 'no limit' and setting to 0. Otherwise set to 1 to indicate restrictions.**  | 'group_size_cap' |
| `columns_to_diff_list` | `list of list` | List of paired features in a inner list used for value differencing purposes. Generalisable to the format of [[...,...], [...,...], ...]. <br /> Example setting the value to `[["minimum_temperature_c","maximum_temperature_c"]]` indicates that feature differencing by `maximum_temperature_c` - `minimum_temperature_c` is to be implemented.|
---
### Additional Feature Engineering: *Adstock features*

| Parameter | Type | Description | Default Value |
| --- | --- | --- | --- |
| `include_adstock` | `bool` | Control that determines if adstock features should be included in feature engineering pipeline process. If set to True, adstock features will be generated, and the user should include all marketing cost features in the `fe_columns_to_drop_list` to prevent correlated features from being input into the model. If False, the parameters below do not apply. | False |
| `adstock_value_threshold` | `float` | Percentage of adstock value on the last campaign date to be used as the cut-off value, for stopping the generation of adstock values beyond the campaign duration. | 0.05 |
| `adstock_days_threshold` | `int` | Number of days beyond the campaign duration to be used as the cut-off value, for stopping the generation of adstock values beyond the campaign duration. | 60 |
| `tv_ad_lag_weight` | `float` | Lag weight for tv_ad  | 0.5 |
| `radio_ad_lag_weight` | `float` | Lag weight for radio_ad | 0.5 |
| `instagram_ad_lag_weight` | `float` | Lag weight for instagram_ad | 0.5 |
| `facebook_ad_lag_weight` | `float` | Lag weight for facebook_ad | 0.5 |
| `youtube_ad_lag_weight` | `float` | Lag weight for youtube_ad | 0.5 |
| `poster_campaign_ad_lag_weight` | `float` | Lag weight for poster_campaign_ad | 0.5 |
| `digital_lag_weight` | `float` | Lag weight for digital_ad | 0.5 |
| `mkt_channel_list` | `list` | List of marketing cost features (represented with `daily_cost` suffix). | `['tv_ad_daily_cost', 'radio_ad_daily_cost', 'instagram_ad_daily_cost', 'facebook_ad_daily_cost', 'youtube_ad_daily_cost','poster_campaign_daily_cost', 'digital_daily_cost']` |

### Time-Dependent Feature Engineering

| Parameter | Type | Description | Default Value |
| --- | --- | --- | --- |
| `fe_target_feature_name` | `str` | Target feature of interest (predicted feature). Primarily referenced by *tsfresh* feature engineering. | 'proxyrevenue' |
| `fe_ordinal_encoding_dict` | `dict` | Dictionary containing columns to be ordinal encoded as key a list of labels as values. Simple example `{"type" : ["carry-out","dine-in"]}`. | {} |
| `fe_one_hot_encoding_col_list` | `list` | List of columns to be one-hot encoded. Example `["type"]`. | ['type'] |
| `binning_dict` | `dict` | Dictionary containing columns to be binned (equal frequncy binning) as keys and bin labels list as values. | { "proxyrevenue" : ['Low', 'Medium', 'High', 'Exceptional'] } |
| `columns_to_std_norm_list` | `list` | List of columns required for either standardisation or normalisation. | [ ] |
| `include_lags_columns_for_std_norm` | `bool` | Control which determines if standardisation/normalization is to be applied for lag features/columns in the dataset. | True |
| `normalization_approach` | `str` | Specified normalization approach to use: either `normalize` or `standardize`. Any other values would be defaulted back to `normalize`. | 'standardize' |
---

### Additional Feature Engineering: *Lag Features Generation*

| Parameter | Type | Description | Default Value |
| --- | --- | --- | --- |
| `columns_to_create_lag_features` | `list` | List of columns which lag features should be created. Parameters for lag generation are controlled in the next 5 parameters below. | `['proxyrevenue']` |
| `lag_periods_list` | `list` | List of lag periods (integers) to generate. This takes into consideration of based on the difference in number of days from the last date of predictions to be made and available data, as per `sma_tsfresh_shift_period` description. | `[9, 14]` |
| `sma_window_periods_list` | `list` | List of window size for simple moving average aggregation. | `[7]` |
| `lag_week_periods_list` | `list` | List of lag periods in terms of weeks to be applied to when weekly average is calculated. | `[1, 2]` |
| `sma_tsfresh_shift_period`  | `int` | Shared *Tsfresh* and *simple moving average* shift periods for the purpose of alignment in shift period. Based on the difference in number of days from the last date of predictions to be made and available data. <br/>Example, on 18th of a month prediction, for 20 to 26th is to be made, while data is only available up to 17th. Hence, there is a 9 days difference which a shift needs to be applied. In view of this, the training process needs to account for such consideration when inference is to be made. | 9 |

### Additional Feature Engineering: *tsfresh*

| Parameter | Type | Description | Default Value |
| --- | --- | --- | --- |
| `include_tsfresh`| `bool` | Control that determines if *tsfresh* should be included in feature engineering pipeline process. If False, the parameters below do not apply. | False |
| `tsfresh_num_features` | `int` | Number of derived *tsfresh* features to use based on a derived list of tsfresh's combined_relevance_tables containing list of tsfresh features for each outlet that satisfies the `tsfresh_n_significant`, based on mean aggregated p-values sorted in ascending order. | 20 |
| `tsfresh_days_per_group` | `int` | Timeshift for *tsfresh* rolling time series' min/max timeshift parameter. Related to `tsfresh_features_list`. | 7 |
| `tsfresh_target_feature` | `str` | Predicted feature to be referenced when applying *tsfresh* feature engineering. | 'binned_proxyrevenue' |
| `tsfresh_features_list` | `list` | Numeric target feature used for *tsfresh* rolling time series which creates subwindows. <br /><br /> Think of it as shifting a cut-out window over your sorted time series data: on each shift step you extract the data you see through your cut-out window to build a new, smaller time series and extract features only on this one. Then you continue shifting. More in https://tsfresh.readthedocs.io/en/latest/api/tsfresh.utilities.html.| ['proxyrevenue'] |
| `tsfresh_n_significant` | `int` | Threshold determining which features should be statistically significant predictors for categorical target feature to be regarded as relevant. <br /><br /> Example, if there are X target categories for prediction, set this value to be <= X | 4 |
---

### Model Specific Preprocessing Module

| Parameter | Type | Description | Default Value |
| --- | --- | --- | --- |
| `target_column_for_modeling` | `str` | Column to serve as predicted model | 'binned_proxyrevenue' |
---

## Key Parameters in `constants.yml`

This sub-section outlines the key parameters in the `constants.yml` configuration file. The file is organised into multiple sections to cater to different modules of the data pipeline.

This file has two intended purposes:
1. Configured values serve as a fallback for invalid parameters set in `parameters.yml` or required parameters that are missing. Note that not all parameters are covered in this `constants.yml`, as it is not feasible to make assumptions on valid values for some parameters.
2. Storage of global constants which are shared.


### Default Configurations for Data Loader

| Parameter | Type | Description | Default Value |
| --- | --- | --- | --- |
| `default_date_col` | `str` | Default column to reference date column based on provided data. It is assumed that such column are represented with a single universal identifier. | 'Date' |
| `default_propensity_factor_column` | `str` | Default column for propensity factor feature. | 'Factor' |
| `default_mkt_channels_column` | `str` | Default column for marketing channels. | 'Mode' |
| `default_mkt_cost_column` | `str` | Default column for marketing channels total cost for each marketing campaign. | 'Total Cost' |
| `default_mkt_name_column` | `str` | Default column for marketing campaigns name. | 'Name' |
| `default_mkt_date_start_end_columns` | `list` | Default columns for marketing's start and end dates. | ['Date Start', 'Date End'] |
| `default_outlet_column` | `str`  | Default column for outlet (aka cost centre codes). | 'CostCentreCode' |
| `columns_to_construct_date` | `dict` | Columns used to construct date feature for different datasets. | - |
| ↳ `weather_data` | `list` | Date related column from weather data used for construction 'Date' feature in the format of `YYYY-MM-DD` format. | ["Year", "Month", "Day"] |
| ↳ `marketing_data` | `list` | Start and end dates for marketing data. | ["Date Start", "Date End"] |
---


### Default Configurations for Data Preprocessing

| Parameter | Type | Description | Default Value |
| --- | --- | --- | --- |
| `default_start_date` | `str` | Default start date for the dataset to be used when invalid format is provided. | '2021-01-01' |
| `default_end_date` | `str` | Default end date for the dataset to be used when invalid format is provided. | '2022-12-31' |
| `default_revenue_column` | `str` | Default column for outlet proxy revenue data. | 'proxyrevenue' |
| `default_const_value_perc_threshold` | `int` | Default constant value percentage threshold to be used when value provided is not within the range of [0,100]. | 0 |
| `default_outlets_exclusion_list` | `list` | Default list of outlets to be excluded. Utilised when the outlets specified to be excluded in parameters is not in a list format. | [ ] |
---

### Default Configurations for Data Split

| Parameter | Type | Description | Default Value |
| --- | --- | --- | --- |
| **Config using dictionary subsection** | | | |
| `data_split` | `dict` | Default configurations for data split. | - |
| ↳ `training_days_default` | `int` | Default number of days allocated for training. | 365 |
| ↳ `validation_days_default` | `int` | Default number of days allocated for validation. | 14 |
| ↳ `testing_days_default` | `int` | Default number of days allocated for testing. | 14 |
| ↳ `window_sliding_stride_days_default` | `int` | Default window stride for sliding window (in days). | 90 |
| ↳ `window_expansion_days_default` | `int` | Default window expansion size for expanding window (in days). | 90  |
| ↳ `simple_split_fold_default` | `int` | Default fold number if "simple_split" is chosen. Note that simple split only produces 1 fold in total. | 1 |
| ↳ `window_split_fold_default`. | `int` | Default fold number if "sliding_window" or "expanding_window" is chosen. | 3 |
| ↳ `data_split_option_list` | `list` | List of available data split option. | ["simple_split", "expanding_window", "sliding_window"] |
| ↳ `data_split_option_default` | `str` | Default data split when invalid option is provided. | "simple_split" |
---

## Key Parameters in `data_split.yml`

This subsection outlines the key parameters in the `data_split.yml` configuration file which handles provides configurable parameters on how different splits involving *simple*, *sliding window* and *expanding window* are implemented. It is broken down into sections, each detailing a specific type of data split.

### Simple Split (under `simple_split`)

| Parameter | Type | Description | Default Value |
| --- | --- | --- | --- |
| `training_days` | `int` | Number of training days. | 588 |
| `testing_days` | `int` | Number of testing days. | 71 |
| `validation_days` | `int` | Number of validation days.  | 71 |
| `window_sliding_stride_days` | `int` | Placeholder for parameters use consistency, DO NOT CHANGE THIS. | 0 |
| `window_expansion_days` | `int` | Placeholder for parameters use consistency, DO NOT CHANGE THIS. | 0 |
| `split_approach` | `str` | Placeholder for parameters use consistency, Splitting approach name. | "simple_split" |
| `folds` | `int` | Placeholder for parameters use consistency, DO NOT CHANGE THIS. | 1 |
---

### Sliding Window Split (under `sliding_window`)

| Parameter | Type | Description | Default Value |
| --- | --- | --- | --- |
| `training_days` | `int` | Number of training days. | 365 |
| `testing_days` | `int` | Number of testing days.  | 60  |
| `validation_days` | `int` | Number of validation days. | 60 |
| `window_sliding_stride_days` | `int` | Days to stride the window. | 90 |
| `window_expansion_days` | `int` | Placeholder for config consistency, DO NOT CHANGE THIS. | 0 |
| `split_approach` | `str` | Splitting approach used | "sliding_window" |
| `folds` | `int` | Number of folds. | 5 |
---

### Expanding Window Split (under `expanding_window`)

| Parameter | Type | Description | Default Value |
| --- | --- | --- | --- |
| `training_days` | `int` | Number of training days. | 365 |
| `testing_days` | `int` | Number of testing days. | 60 |
| `validation_days` | `int` | Number of validation days. | 60 |
| `window_sliding_stride_days` | `int` | Placeholder for config consistency, DO NOT CHANGE THIS. | 0 |
| `window_expansion_days` | `int` | Days set for expanding window for each fold. | 90 |
| `split_approach` | `str` | Splitting approach used. | "expanding_window"|
| `folds` | `int` | Number of folds required for split. | 5 |
---

## Executing the Data Pipeline on a Docker Container (Windows only)

This guide explains how to run the data pipeline using the `run_data_pipeline.bat` script on a Windows OS environment. The script offers an automated and convenient way to trigger the pipeline processes. Before executing the script, please ensure that you have loaded the provided docker image on the host machine. This can be done via the following command:

```
  # windows powershell
  docker load -i <path/to>/100E_BIPO_docker_training.tar
```

### Pushing Docker image to a preferred registry

In the event you would need to push the Docker image to your internal registry, please rename your Docker image to the source URL of the Docker registry as the existing registry used points to `registry.aisingapore.net/100e-bipo`, which would not be accessible outside of AI Singapore. Please refer to the [Docker documentation](https://docs.docker.com/) for details on the renaming procedure to facilitate push to a Docker registry.

### Usage of `run_data_pipeline.bat`

Key configurations in the file are described below. In the event that you have pushed the provided Docker image into your registry, amend only the following variables `docker_registry` and `image_name` in the script as they are related to the registry URL and image name. 

Leave `docker_registry` variable blank if this is not applicable. The script will use the loaded Docker image for running.

```
@REM Define your Docker image and registry
set docker_registry=<Url to docker registry>/<project folder>
set image_name=<image name>
```

The following governs the directories to be mounted.

```
@REM Define volume variables from host directories
set DATA_DIR=data
set CONF_DIR=conf
set LOGS_DIR=logs
```

### Running `run_data_pipeline.bat`

To initiate the pipeline:

1. Open Windows Powershell.
2. Navigate to the project root directory `bipo_demand_forecasting`.
3. Execute the following command:
    ```
    .\scripts\run_data_pipeline.bat
    ```

### Logging

Logs provide essential insights into the runtime behaviour of your pipeline, enabling effective monitoring the pipeline's execution. Generated log files are located in `logs` folder at the project parent directory. 

For log settings, please navigate to `conf/base/logging.yml` to customise according to your needs. 

> **Note:**
> Do not delete this file before running the pipeline or Kedro will throw an error.

The log filepath used in the yml file is a placeholder and would be created and deleted when a Kedro pipeline starts and completes its execution. Nothing important is logged to the file since a separate log file (in daily format) is generated instead. This is implemented via the script `src/bipo/hooks/SetupDailyLogsHooks.py`, serving as Kedro Hook's mechanism which overwrites the specified logfile path with today's date whenever a pipeline is run.

The following levels of logs are used in the code base:
- Info
- Error
- Debug (Included as log handler, unused. Intended for development purposes.)

#### Successful start of Kedro pipeline run

The log snippet below shows a successful start of a Kedro pipeline run:
```
18/10/2023 08:24 | kedro.io.data_catalog | INFO | Loading data from 'loaded_non_proxy_revenue_partitioned_data' (IncrementalDataSet)...
18/10/2023 08:24 | kedro.pipeline.node | INFO | Running node: data_preprocessing_merge_non_proxy_revenue_data: merge_non_proxy_revenue_data([loaded_non_proxy_revenue_partitioned_data]) -> [merged_non_revenue_data]
18/10/2023 08:24 | kedro | INFO | Loading partition: marketing_restructured
18/10/2023 08:24 | kedro | INFO | Using marketing_restructured as base dataframe.
18/10/2023 08:24 | kedro | INFO | Loading partition: merged_unique_daily_records
18/10/2023 08:24 | kedro | INFO | Preparing merge with partition: merged_unique_daily_records
18/10/2023 08:24 | kedro | INFO | Merged partition: merged_unique_daily_records to existing dataframe.

18/10/2023 08:24 | kedro | INFO | Loading partition: propensity_restructured
18/10/2023 08:24 | kedro | INFO | Preparing merge with partition: propensity_restructured
18/10/2023 08:24 | kedro | INFO | Merged partition: propensity_restructured to existing dataframe.

18/10/2023 08:24 | kedro | INFO | Loading partition: weather_restructured
18/10/2023 08:24 | kedro | INFO | Preparing merge with partition: weather_restructured
18/10/2023 08:24 | kedro | INFO | Merged partition: weather_restructured to existing dataframe.

18/10/2023 08:24 | kedro | INFO | Completed merging of non proxy revenue dataframe. Dataframe is of shape (1099, 24)

18/10/2023 08:24 | kedro.io.data_catalog | INFO | Saving data to 'merged_non_revenue_data' (MemoryDataSet)...
18/10/2023 08:24 | kedro.runner.sequential_runner | INFO | Completed 1 out of 43 tasks
18/10/2023 08:24 | kedro.io.data_catalog | INFO | Loading data from 'loaded_proxy_revenue_partitioned_data' (IncrementalDataSet)...
18/10/2023 08:24 | kedro.io.data_catalog | INFO | Loading data from 'parameters' (MemoryDataSet)...
18/10/2023 08:24 | kedro.pipeline.node | INFO | Running node: feature_engr_generate_lag: generate_lag([loaded_proxy_revenue_partitioned_data,parameters]) -> [lag_features_partitions_dict]
18/10/2023 08:24 | kedro | INFO | Column specified for lag generation {'proxyrevenue'}
...
```

#### Successful completion of Kedro pipeline run

The snippet below is an illustration of a successful completion of Kedro pipeline run (containing the wording `Pipeline execution completed successfully`): 

```
...
25/10/2023 09:17 | kedro.io.data_catalog | INFO | Saving data to 'ebm.model_specific_preprocessing_training' (IncrementalDataSet)...
25/10/2023 09:17 | kedro.io.data_catalog | INFO | Saving data to 'ebm.model_specific_preprocessing_validation' (IncrementalDataSet)...
25/10/2023 09:17 | kedro.io.data_catalog | INFO | Saving data to 'ebm.model_specific_preprocessing_testing' (MemoryDataSet)...
25/10/2023 09:17 | kedro.runner.sequential_runner | INFO | Completed 49 out of 49 tasks
25/10/2023 09:17 | kedro.runner.sequential_runner | INFO | Pipeline execution completed successfully.
```