# Data Pipeline

## Overview
This guide outlines the architecture and components of the training data pipeline, designed to be modular, flexible, and easily adaptable to various data sources and machine learning models. It is divided into specific modules for different data preparation and processing tasks.

## Data Sources

| Filename                              | Description                                                                | Source                                          |
| ------------------------------------- | -------------------------------------------------------------------------- | ----------------------------------------------- |
| `proxy_revenue_masked.csv`            | Daily proxy revenue of all outlets.                                        | BIPO                                            |
| `marketing cost.xlsx`                 | Cost breakdown of marketing campaigns by mode.                             | BIPO                                            |
| `consumer propensity to spend.xlsx`   | Daily consumer propensity to spend by regions in Singapore.                | BIPO                                            |
| `SG climate records 2021 - 2022.xlsx` | Daily climate data from four key regions in Singapore.                     | BIPO                                            |
| `holiday_df.xlsx`                     | Mapping of past dates to school holidays and public holidays in Singapore. | MOE Website, School Terms and Holidays for 2022 |

## Design Architecture
Design Considerations:
1. Modular Design: Each component is modular for easy maintenance and updates.
2. Decoupling: Pipeline stages are designed to work independently, allowing for easier testing and modifications.
3. Configurability: Most components are configurable through config files, allowing for easy adjustments.

The diagram below provides a high-level overview of the data pipeline's architecture. 

![Pipeline Design](./assets/data_pipeline_training.png)

It outlines the transformation and processing steps for raw data in preparation for model training.

| **Step**                           | **Activity**                                                                                                                                                                                | **Configuration File**            | **Output Store**        |
| ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------- | ----------------------- |
| Raw Data                           | Initial data is sourced from a variety of structured formats, including `.csv` and `.xlsx` files. For a detailed list of these sources, refer to the [Data Sources section](#data-sources). | -                                 | -                       |
| Data Loading & Validation          | Ingest data using configurations and verify its quality.                                                                                                                                    | `constants.yml`, `parameters.yml`  | Validated Data          |
| Data Preprocessing                 | Process and cleanse raw data to make it suitable for subsequent steps.                                                                                                                      | `constants.yml`, `parameters.yml`  | Preprocessed Data       |
| Data Merge                         | Combine various preprocessed data sources.                                                                                                                                                  | `constants.yml`                   | Merged Outlet Data      |
| Data Split                         | Divide merged data into different subsets (like training, validation, and test).                                                                                                            | `constants.yml`, `parameters.yml`  | Split Data              |
| Time Agnostic Feature Engineering  | Engineer data features that are not dependent on time.                                                                                                                                      | `constants.yml`, `parameters.yml`  | Feature Engineered Data |
| Time Dependent Feature Engineering | Engineer data features that take time factors into account.                                                                                                                                 | `constants.yml`, `parameters.yml` | Feature Engineered Data |
| Model-specific Preprocessing       | Further process data based on specific requirements of the target model.                                                                                                                    | `constants.yml`, `parameters.yml`  | Model Input Data        |
  
## 1. Data Loading & Validation
This module focuses on ingesting and validating data from multiple sources for reliable use in later stages.

**Diagram**

Refer to the included visual representation for a structured view of this process.

![Pipeline Design](./assets/data_loader.png)

### Input
Gathered raw data awaiting cleaning and structuring.

| Component                        | Description                                                     |
| -------------------------------- | --------------------------------------------------------------- |
| raw_propensity_data              | Customer propensity data in an Excel format.                    |
| raw_weather_data                 | Weather-related data in an Excel format.                        |
| raw_marketing_data               | Marketing information in an Excel format.                       |
| raw_proxy_revenue_data           | Proxy revenue figures also presented in an Excel format.        |
| xlsx_raw_unique_daily_partitions | Folder with daily Excel files, such as `holiday_df.xlsx`.       |
| csv_raw_unique_daily_partitions  | Folder with daily CSV files, for example, `covid_capacity.csv`. |

### Output
Processed datasets, prepped for analysis.

| Component                             | Description                                                        | Type               |
| ------------------------------------- | ------------------------------------------------------------------ | ------------------ |
| loaded_propensity_data                | Propensity data made ready for analysis.                           | CSVDataSet         |
| loaded_weather_data                   | Weather data that's cleaned and organised.                         | CSVDataSet         |
| loaded_marketing_data                 | Marketing data that's prepared for extracting insights.            | CSVDataSet         |
| loaded_proxy_revenue_partitioned_data | Segmented revenue data made ready for detailed analysis.           | PartitionedDataSet |
| merged_xlsx                           | Combined daily records from XLSX files.                            | MemoryDataset      |
| merged_csv                            | Combined daily records from CSV files.                             | MemoryDataset      |
| merged_unique_daily_temp              | Combined records by date, prepared for the final processing steps. | MemoryDataset      |
| merged_unique_daily                   | Daily records sorted by date for time-based analysis.              | CSVDataSet         |



## 2. Data Preprocessing
The data_preprocessing module combines *non-proxy revenue datasets* with *individual outlet proxy revenue datasets*. The pipeline employs sanity checks like date validity, outlet exclusion, and constant value thresholds.

**Diagram**

Refer to the included visual representation for a structured view of this process.

![Pipeline Design](./assets/data_preprocessing.png)

### Input
Collected data from various categories are inspected and amalgamated.

| Component                                 | Description                                                                          |
| ----------------------------------------- | ------------------------------------------------------------------------------------ |
| loaded_non_proxy_revenue_partitioned_data | Folder with partition IDs and pointers for loading non-proxy revenue data.       |
| loaded_proxy_revenue_partitioned_data     | Folder with partition IDs and pointers for loading outlet-specific revenue data. |

### Output
Data from various partitions are collected and checked for subsequent analytics.

| Component               | Description                                                               | Type               |
| ----------------------- | ------------------------------------------------------------------------- | ------------------ |
| merged_non_revenue_data | Consolidated DataFrame of non-proxy revenue data from various partitions. | MemoryDataset      |
| data_preprocessed       | Folder of processed outlets partition.                                | PartitionedDataSet |

## 3. Data Merging & Splitting
The data_splitting module is responsible for merging the preprocessed datasets and then segmenting the them into training, validation, and test sets. This ensures that the model can be trained and evaluated effectively. 

**Diagram**

Refer to the included visual representation for a structured view of this process. 

![Pipeline Design](./assets/data_splitting.png)

### Input
The cleaned, validated, and combined dataset ready for splitting into training, validation, and test sets.

| Component         | Description                                                        |
| ----------------- | ------------------------------------------------------------------ |
| data_preprocessed | Processed outlets partition, ready for data merging and splitting. |

This data is then passed through different data splitting approaches like `simple_split`, `expanding_window`, and `sliding_window`, as specified in the configuration file.

### Output
Datasets partitioned for model training, validation, and testing. Different techniques like simple_split, expanding_window, and sliding_window could be used based on pipeline configuration. 

| Component  | Description                                                                                      | Type               |
| ---------- | ------------------------------------------------------------------------------------------------ | ------------------ |
| data_merge | A dataframe that consolidates outlets specified files into one, sorted by a default date column. | CSVDataSet         |
| data_split | Partitioned datasets containing separate subsets for training, validation, and testing.          | PartitionedDataSet |

## 4. Feature Engineering
The Feature Engineering module boosts model accuracy by crafting advanced features from your data. It uses statistical metrics, lagging, categorical encoding, and libraries like tsfresh and LightweightMMM.

### Feature Inventory
Here are the features engineered by this module, mapped to their respective data sources. Each entry includes the feature's name, a brief description, and an indication of whether or not the feature is utilised in the final model training.

| Feature Source                        | Feature Name                           | Description                                                          | Included in Trained Model |
| ------------------------------------- | -------------------------------------- | -------------------------------------------------------------------- | ------------------------- |
| Native Python Library              | `is_weekend`                           | Indicates if the day is a weekend                                    | Yes                       |
| `holiday_df.xlsx`                     | `is_public_holiday`                    | Signifies if the day is a public holiday                             | Yes                       |
| `holiday_df.xlsx`                     | `is_school_holiday`                    | Specifies if the day is a school holiday                             | Yes                       |
| `proxy_revenue_masked.csv`            | `binned_proxy_revenue`                 | Categorised proxy revenue data                                       | Yes                       |
| Native Dataset / Library              | `day_of_week`                          | Day of the week                                                      | Yes                       |
| Native Dataset / Library              | `is_pandemic_restrictions`             | Indicates presence of pandemic-related restrictions                  | Yes                       |
| `SG climate records 2021 - 2022.xlsx` | `is_raining`                           | Denotes if it's a rainy day                                          | Yes                       |
| `SG climate records 2021 - 2022.xlsx` | `temp_max`                             | Maximum temperature for the day                                      | Yes                       |
| `proxy_revenue_masked.csv`            | `lag_9_days_proxy_revenue`             | Revenue data lagged by 9 days                                        | Yes                       |
| `proxy_revenue_masked.csv`            | `lag_14_days_proxy_revenue`            | Revenue data lagged by 14 days                                       | Yes                       |
| `proxy_revenue_masked.csv`            | `sma_window_7_days_proxy_revenue`      | 7-day Simple Moving Average of proxy revenue                         | Yes                       |
| `proxy_revenue_masked.csv`            | `sma_window_8_days_proxy_revenue`      | 8-day Simple Moving Average of proxy revenue                         | Yes                       |
| `proxy_revenue_masked.csv`            | `lag_1_week_mean_weekly_proxy_revenue` | Mean weekly proxy revenue lagged by 1 week                           | Yes                       |
| `proxy_revenue_masked.csv`            | `lag_2_week_mean_weekly_proxy_revenue` | Mean weekly proxy revenue lagged by 2 weeks                          | Yes                       |
| `marketing cost.xlsx`                 | `cat_mkt_campaign_start`               | Start date of ongoing marketing campaigns                            | Yes                       |
| `marketing cost.xlsx`                 | `cat_mkt_campaign_end`                 | End date of ongoing marketing campaigns                              | Yes                       |
| `marketing cost.xlsx`                 | `count_mkt_campaign`                   | Number of ongoing marketing campaigns for the current date           | Yes                       |
| `marketing cost.xlsx`                 | `is_having_campaign`                   | Indicates if any marketing campaigns are active for the current date | Yes                       |
| `marketing cost.xlsx`                 | `radio_ad_daily_cost`                  | Daily cost for radio advertisements                                  | Yes                       |
| `marketing cost.xlsx`                 | `digital_daily_cost`                   | Daily cost for digital marketing                                     | Yes                       |
| `marketing cost.xlsx`                 | `instagram_ad_daily_cost`              | Daily cost for Instagram advertisements                              | Yes                       |
| `marketing cost.xlsx`                 | `poster_campaign_daily_cost`           | Daily cost for poster campaigns                                      | Yes                       |
| `marketing cost.xlsx`                 | `tv_ad_daily_cost`                     | Daily cost for TV advertisements                                     | Yes                       |
| `marketing cost.xlsx`                 | `youtube_ad_daily_cost`                | Daily cost for YouTube advertisements                                | Yes                       |
| `marketing cost.xlsx`                 | `facebook_ad_daily_cost`               | Daily cost for Facebook advertisements                               | Yes                       |
| `marketing cost.xlsx`                 | `campaign_daily_cost`                  | Overall daily cost for all marketing campaigns                       | Yes                       |
| Lightweight MMM                       | `adstock_<channel>_daily_cost`         | Adstock effect on daily cost for a given channel                     | No                        |
| Lightweight MMM                       | `carryover_<channel>_daily_cost`       | Carryover effect on daily cost for a given channel                   | No                        |


## 5. Model-specific Preprocessing
This module is geared towards *customised preprocessing steps* that are specifically *tailored for the selected machine learning model in use*. Depending on the model selected, this can include tasks like feature scaling, encoding categorical variables, or even more complex data transformations. 

**Diagram**

Refer to the included visual representation for a structured view of this process.

![Pipeline Design](./assets/data_model_specific_preprocessing.png)

### Input
Feature-engineered data customised for the specific predictive models being used in the analysis.

| Component                                          | Description                                                                                                                        |
| -------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| feature_engineering_with_lightweightmmm_training   | Feature-engineered data optimised for training the LightweightMMM model. Includes transformations and variable selection.          |
| feature_engineering_with_lightweightmmm_validation | Feature-engineered data optimised for validating the LightweightMMM model. Similar transformations applied as in the training set. |

### Output
Data that has been specifically preprocessed to suit the requirements of the chosen predictive model(s). This includes creating concatenated folds, removing unimportant columns, dealing with constant columns, and other model-specific transformations.

| Component                                       | Description                                                                                                                | Type               |
| ----------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- | ------------------ |
| concatenated_folds_training                     | Training data folds concatenated for model optimisation.                                                                   | MemoryDataset      |
| concatenated_folds_validation                   | Validation data folds concatenated for model evaluation.                                                                   | MemoryDataset      |
| removed_columns_training                        | Training data with irrelevant or less significant columns removed.                                                         | MemoryDataset      |
| removed_columns_validation                      | Validation data with irrelevant or less significant columns removed.                                                       | MemoryDataset      |
| constant_column_params                          | Settings used to identify and remove constant columns across datasets.                                                     | JSONDataSet        |
| remove_const_colum_data_training                | Training data with constant columns removed based on constant_column_params.                                               | MemoryDataset      |
| remove_const_colum_data_validation              | Validation data with constant columns removed based on constant_column_params                                              | MemoryDataset      |
| reordered_data_folds                            | Ordered dictionary that pairs sorted training and validation data partitions by outlet folds for subsequent processing.    | MemoryDataset      |
| {model}.model_specific_preprocessing_train      | Model-specific transformations applied on the training set, formatted according to the specific requirements of {model}.   | PartitionedDataSet |
| {model}.model_specific_preprocessing_validation | Model-specific transformations applied on the validation set, formatted according to the specific requirements of {model}. | PartitionedDataSet |
| {model}.model_specific_preprocessing_test       | Model-specific transformations applied on the test set, formatted according to the specific requirements of {model}        | PartitionedDataSet |

## Data Pipeline Configuration
This section captures some of the key parameters from the `parameters.yml`, `constants.yml` and `data_split.yml` files. Due to the volume of parameters, we'll only include a selection for illustrative purposes.

### Key Parameters in the `parameters.yml` File

This sub-section outlines the key parameters in the `parameters.yml` configuration file that control different aspects of a data pipeline. The file is divided into several sections, each dealing with a specific part of the pipeline. Here's a breakdown:

**Dataloader Configuration**

| Parameter            | Type  | Description        | Default Value    |
| -------------------- | ----- | ------------------ | ---------------- |
| `outlet_column_name` | `str` | Outlet column name | 'costcentrecode' |

**Data Preprocess Configurations**

| Parameter                 | Type   | Description                                   | Default Value |
| ------------------------- | ------ | --------------------------------------------- | ------------- |
| `start_date`              | `str`  | Data start date                               | '2021-01-01'  |
| `end_date`                | `str`  | Data end date                                 | '2022-12-31'  |
| `zero_val_threshold_perc` | `int`  | Percentage of zero values allowed in a column | 2             |
| `outlets_exclusion_list`  | `list` | List of outlets to be excluded                | []            |

**Feature Engineering Configurations: Time Agnostic** 

| Parameter                       | Type   | Description                                      | Default Value                                                                                                                                                                                                                                                                  |
| ------------------------------- | ------ | ------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `fe_columns_to_drop_list`       | `list` | Columns to be dropped before feature engineering | ['day', 'location', 'manhours', 'outlet', 'highest_30_min_rainfall_mm', 'highest_60_min_rainfall_mm', 'highest_120_min_rainfall_mm', 'mean_temperature_c', 'minimum_temperature_c', 'mean_wind_speed_kmh', 'max_wind_speed_kmh', 'school_holiday_type', 'public_holiday_type'] |
| `fe_mkt_column_name`            | `str`  | Column name for marketing-related features       | 'name'                                                                                                                                                                                                                                                                         |
| `fe_mkt_columns_to_impute_dict` | `dict` | Default values for marketing-related columns     | {'tv_ad_daily_cost': 0, 'radio_ad_daily_cost': 0, 'instagram_ad_daily_cost': 0, 'facebook_ad_daily_cost': 0, 'youtube_ad_daily_cost': 0, 'poster_campaign_daily_cost': 0, 'digital_daily_cost': 0}                                                                             |

**Feature Engineering Configurations: Time-Agnostic Features Categorisation**

| Parameter               | Type  | Description                    | Default Value  |
| ----------------------- | ----- | ------------------------------ | -------------- |
| `split_approach_source` | `str` | Method used for data splitting | 'simple_split' |

**Feature Engineering Configurations: Column Specific Features**

| Parameter                | Type   | Description                                  | Default Value                        |
| ------------------------ | ------ | -------------------------------------------- | ------------------------------------ |
| `fe_rainfall_column`     | `str`  | Column used for 'is_raining' feature         | 'daily_rainfall_total_mm'            |
| `fe_holiday_column_list` | `list` | Columns for generating holiday boolean state | ['school_holiday', 'public_holiday'] |
| `fe_pandemic_column`     | `str`  | Column for generating pandemic boolean state | 'group_size_cap'                     |

**Time-Dependent Feature Engineering**

| Parameter                           | Type   | Description                                              | Default Value                            |
| ----------------------------------- | ------ | -------------------------------------------------------- | ---------------------------------------- |
| `fe_target_feature_name`            | `str`  | Target feature of interest                               | 'proxyrevenue'                           |
| `fe_ordinal_encoding_dict`          | `dict` | Dictionary containing columns to encode and their labels | {}                                       |
| `fe_one_hot_encoding_col_list`      | `list` | Columns to apply one-hot encoding                        | ['type']                                 |
| `binning_dict`                      | `dict` | Binning dictionary for target feature                    | ['Low', 'Medium', 'High', 'Exceptional'] |
| `fe_columns_to_std_norm_list`       | `list` | Columns to apply standard normalization                  | []                                       |
| `include_lags_columns_for_std_norm` | `bool` | Include lag columns for standard normalization           | True                                     |
| `normalization_approach`            | `str`  | Normalization approach to use                            | 'standardize'                            |

**Additional Feature Engineering**

| Parameter                        | Type   | Description                    | Default Value                                                                                                                                                                |
| -------------------------------- | ------ | ------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `columns_to_create_lag_features` | `list` | Columns to create lag features | ['proxyrevenue']                                                                                                                                                             |
| `lag_periods_list`               | `list` | List of lag periods            | [9, 14]                                                                                                                                                                      |
| `sma_window_periods_list`        | `list` | SMA window periods list        | [7]                                                                                                                                                                          |
| `sma_shift_period`               | `int`  | SMA shift period               | 9                                                                                                                                                                            |
| `lag_week_periods_list`          | `list` | Lag week periods list          | [1, 2]                                                                                                                                                                       |
| `lightweightmmm_num_lags`        | `int`  | Lightweight MMM number of lags | 7                                                                                                                                                                            |
| `fe_mkt_channel_list`            | `list` | FE market channel list         | ['tv_ad_daily_cost', 'radio_ad_daily_cost', 'instagram_ad_daily_cost', 'facebook_ad_daily_cost', 'youtube_ad_daily_cost','poster_campaign_daily_cost', 'digital_daily_cost'] |
| `include_lightweightMMM`                | `bool`     | Include tsfresh in pipeline                          | False                                                                                                                 |
| `lightweightmmm_adstock_normalise`      | `bool`     | Normalize adstock values                             | True                                                                                                                  |
| `lightweightmmm_optimise_parameters`    | `bool`     | Optimize lightweightmmm parameters                   | True                                                                                                                  |
| `lightweightmmm_params`                 | `dict`     | Parameters for lightweightmmm                        | Various                                                                                                              |
| ↳ `lag_weight`                          | `list`     | Lag weight                                           | [0.7025, 0.9560, 0.7545, 0.7484, 0.9405, 0.6504, 0.7207]                                                             |
| ↳ `ad_effect_retention_rate`            | `list`     | Ad effect retention rate                             | [0.4963, 0.6495, 0.5482, 0.5484, 0.5822, 0.6404, 0.7346]                                                             |
| ↳ `peak_effect_delay`                   | `list`     | Peak effect delay                                   | [1.2488, 5.5658, 1.9455, 1.8988, 1.8554, 1.6911, 1.8539]                                                             |                                                                                                                                           |


**Tsfresh Related Features**

| Parameter                   | Type   | Description                    | Default Value                                                                                                                                                                                                    |
| --------------------------- | ------ | ------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `include_tsfresh`           | `bool` | Include TSFresh in pipeline    | False                                                                                                                                                                                                            |
| `run_tsfresh_fe`            | `bool` | Run TSFresh feature extraction | False                                                                                                                                                                                                            |
| `tsfresh_feature_selection` | `bool` | TSFresh feature selection      | True                                                                                                                                                                                                             |
| `tsfresh_entity`            | `str`  | TSFresh entity                 | 'week_sma'                                                                                                                                                                                                       |
| `tsfresh_num_features`      | `int`  | TSFresh number of features     | 20                                                                                                                                                                                                               |
| `tsfresh_days_per_group`    | `int`  | TSFresh days per group         | 7                                                                                                                                                                                                                |
| `tsfresh_target_feature`    | `str`  | TSFresh target feature         | 'binned_proxyrevenue'                                                                                                                                                                                            |
| `tsfresh_features_list`     | `list` | TSFresh features list          | ['proxyrevenue']                                                                                                                                                                                                 |
| `tsfresh_extract_relevant`  | `bool` | TSFresh extract relevant       | True                                                                                                                                                                                                             |
| `tsfresh_n_significant`     | `int`  | TSFresh N significant          | 4                                                                                                                                                                                                                |
| `tsfresh_num_outlets`       | `int`  | TSFresh number of outlets      | 3                                                                                                                                                                                                                |
| `sma_tsfresh_shift_period`  | `int`  | Shared TSFresh and SMA shift period | 9                                                                                                                                                                                                             |


**Model Preprocessing Module**

| Parameter                                | Type   | Description                            | Default Value         |
| ---------------------------------------- | ------ | -------------------------------------- | --------------------- |
| `model`                                  | `str`  | Model to use                           | 'ebm'                 |
| `target_column_for_modeling`             | `str`  | Target column for modeling             | 'binned_proxyrevenue' |
| `drop_columns_used_for_binning_encoding` | `bool` | Drop columns used for binning/encoding | True                  |
| `training_testing_mode`                  | `str`  | Training or testing mode               | 'training'            |
| `fold`                                   | `int`  | Fold number                            | 1                     |

**MLflow Tracking Server**

| Parameter                | Type   | Description            | Default Value               |
| ------------------------ | ------ | ---------------------- | --------------------------- |
| `enable_mlflow`          | `bool` | Enable MLFlow          | False                       |
| `is_remote_mlflow`       | `bool` | Is MLFlow remote       | False                       |
| `tracking_uri`           | `str`  | Tracking URI           | 'http://10.43.130.112:5005' |
| `experiment_name_prefix` | `str`  | Experiment name prefix | 'bipo'                      |


### Key Parameters in the `constants.yml` File

This sub-section outlines the key parameters in the `constants.yml` configuration file. The file is organised into multiple sections to cater to different aspects of data loading, preprocessing, and modelling.

**Default Configurations for Dataloader**

| Parameter                            | Type   | Description                                       | Default Value              |
| ------------------------------------ | ------ | ------------------------------------------------- | -------------------------- |
| `default_date_col`                   | `str`  | Default column for date                           | 'Date'                     |
| `default_propensity_factor_column`   | `str`  | Default column for propensity factor              | 'Factor'                   |
| `default_mkt_channels_column`        | `str`  | Default column for marketing channels             | 'Mode'                     |
| `default_mkt_cost_column`            | `str`  | Default column for marketing total cost           | 'Total Cost'               |
| `default_mkt_name_column`            | `str`  | Default column for marketing campaigns name       | 'Name'                     |
| `default_mkt_date_start_end_columns` | `list` | Default columns for marketing start and end dates | ['Date Start', 'Date End'] |
| `default_outlet_column`              | `str`  | Default column for outlet or cost centre          | 'CostCentreCode'           |
| `columns_to_construct_date`          | `dict` | Columns to construct date for different datasets  | Various                    |
| ↳ `weather_data`                     | `list` | Dates for weather data                            | ["Year", "Month", "Day"]   |
| ↳ `marketing_data`                   | `list` | Start and end dates for marketing data            | ["Date Start", "Date End"] |

**Default Configurations for Data Preprocessing**

| Parameter                            | Type   | Description                                 | Default Value  |
| ------------------------------------ | ------ | ------------------------------------------- | -------------- |
| `default_start_date`                 | `str`  | Default start date for the dataset          | '2021-01-01'   |
| `default_end_date`                   | `str`  | Default end date for the dataset            | '2022-12-31'   |
| `default_revenue_column`             | `str`  | Default column for revenue                  | 'proxyrevenue' |
| `default_const_value_perc_threshold` | `int`  | Default constant value percentage threshold | 0              |
| `default_outlets_exclusion_list`     | `list` | Default list of outlets to be excluded      | []             |

**Default Configurations Marketing Columns Generation**

| Parameter                         | Type   | Description                               | Default Value                                                                                                                                                                |
| --------------------------------- | ------ | ----------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `default_marketing_channel_list`  | `list` | Default marketing channel list            | ['tv_ad_daily_cost', 'radio_ad_daily_cost', 'instagram_ad_daily_cost', 'facebook_ad_daily_cost', 'youtube_ad_daily_cost','poster_campaign_daily_cost', 'digital_daily_cost'] |
| `default_lightweightmmm_num_lags` | `int`  | Default number of lags for lightweightmmm | 7                                                                                                                                                                            |

**Default Configurations for Data Splitting Strategy**

| Parameter                              | Type   | Description                                                                       | Default Value                                          |
| -------------------------------------- | ------ | --------------------------------------------------------------------------------- | ------------------------------------------------------ |
| `data_split`                           | `dict` | Default configurations for data splitting                                         | Various                                                |
| ↳ `training_days_default`              | `int`  | Default number of days allocated for training                                     | 365                                                    |
| ↳ `validation_days_default`            | `int`  | Default number of days allocated for validation                                   | 0                                                      |
| ↳ `testing_days_default`               | `int`  | Default number of days allocated for testing                                      | 14                                                     |
| ↳ `window_sliding_stride_days_default` | `int`  | Default stride for sliding window (in days)                                       | 90                                                     |
| ↳ `window_expansion_days_default`      | `int`  | Default expansion size for expanding window (in days)                             | 90                                                     |
| ↳ `simple_split_fold_default`          | `int`  | Default fold number if "simple_split" is chosen                                   | 1                                                      |
| ↳ `window_split_fold_default`          | `int`  | Default fold number if "sliding_window" or "expanding_window" is chosen           | 3                                                      |
| ↳ `data_split_option_default`          | `str`  | Default splitting strategy ("simple_split", "expanding_window", "sliding_window") | "simple_split"                                         |
| ↳ `data_split_option_list`             | `list` | List of available splitting strategies                                            | ["simple_split", "expanding_window", "sliding_window"] |
| ↳ `data_split_fold_default`            | `int`  | Default fold number for any data splitting option                                 | 1                                                      |

**Default Configurations for Model-Specific Preprocessing**

| Parameter                         | Type    | Description                                                     | Default Value                                            |
| --------------------------------- | ------- | --------------------------------------------------------------- | -------------------------------------------------------- |
| `modeling`                        | `dict`  | Root dictionary for model-specific configurations               | Various                                                  |
| ↳ `training_testing_mode_default` | `str`   | Default mode for training or testing                            | "training"                                               |
| ↳ `valid_training_testing_modes`  | `list`  | List of valid modes for training and testing                    | ["training", "testing"]                                  |
| ↳ `model_name_default`            | `str`   | Default name for the predictive model                           | "ebm"                                                    |
| ↳ `valid_model_name`              | `list`  | List of valid model names                                       | ["ordered_model", "ebm"]                                 |
| ↳ `ebm`                           | `dict`  | Default parameters for EBM (Explainable Boosting Machine) model | Various                                                  |
| ↳↳ `outer_bags`                   | `int`   | Number of outer bags for EBM                                    | 10                                                       |
| ↳↳ `inner_bags`                   | `int`   | Number of inner bags for EBM                                    | 0                                                        |
| ↳↳ `learning_rate`                | `float` | Learning rate for EBM                                           | 0.01                                                     |
| ↳↳ `interactions`                 | `int`   | Number of interactions for EBM                                  | 0                                                        |
| ↳↳ `max_leaves`                   | `int`   | Maximum number of leaves for EBM                                | 3                                                        |
| ↳↳ `min_samples_leaf`             | `int`   | Minimum samples per leaf for EBM                                | 2                                                        |
| ↳↳ `max_bins`                     | `int`   | Maximum number of bins for EBM                                  | 256                                                      |
| ↳ `ordered_model`                 | `dict`  | Default parameters for Ordered Model                            | Various                                                  |
| ↳↳ `const_col_artefacts_path`     | `str`   | Path for constant column artefacts                              | "data/06_model_specific_preprocessing/ordered_model.pkl" |
| ↳↳ `distr`                        | `str`   | Distribution type for Ordered Model                             | "probit"                                                 |
| ↳↳ `method`                       | `str`   | Optimization method for Ordered Model                           | "bfgs"                                                   |
| ↳↳ `max_iter`                     | `int`   | Maximum iterations for Ordered Model                            | 2                                                        |

**Dataloader Specifics**

| Parameter                | Type   | Description                                                  | Default Value                                                                                                                                                                                                                                                                                                                                          |
| ------------------------ | ------ | ------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `dataloader`             | `dict` | Parent key for Dataloader configurations                     | Various                                                                                                                                                                                                                                                                                                                                                |
| ↳ `expected_columns`     | `dict` | Specifies the columns expected in each type of input dataset | Various                                                                                                                                                                                                                                                                                                                                                |
| ↳↳ `proxy_revenue_data`  | `list` | List of expected columns in the proxy revenue data           | ["CostCentreCode", "Date", "ManHours", "ProxyRevenue", "Location", "Outlet", "Type"]                                                                                                                                                                                                                                                                   |
| ↳↳ `propensity_data`     | `list` | List of expected columns in the propensity data              | ["Location", "Factor"]                                                                                                                                                                                                                                                                                                                                 |
| ↳↳ `marketing_data`      | `list` | List of expected columns in the marketing data               | ["Name", "Date Start", "Date End", "Mode", "Total Cost"]                                                                                                                                                                                                                                                                                               |
| ↳↳ `weather_data`        | `list` | List of expected columns in the weather data                 | ["DIRECTION", "Station", "Year", "Month", "Day", "Daily Rainfall Total (mm)", "Highest 30 min Rainfall (mm)", "Highest 60 min Rainfall (mm)", "Highest 120 min Rainfall (mm)", "Mean Temperature (°C)", "Maximum Temperature (°C)", "Minimum Temperature (°C)", "Mean Wind Speed (km/h)", "Max Wind Speed (km/h)"]                                     |
| ↳↳ `covid_capacity_data` | `list` | List of expected columns in the COVID capacity data          | ["Group size Cap "]                                                                                                                                                                                                                                                                                                                                    |
| ↳↳ `holiday_data`        | `list` | List of expected columns in the holiday data                 | ["Date", "School Holiday", "School Holiday Type", "Public Holiday", "Public Holiday Type", "Day"]                                                                                                                                                                                                                                                      |
| ↳↳ `merged_df`           | `list` | List of expected columns in the merged dataframe             | ["cost_centre_code", "man_hours", "proxy_revenue", "location", "outlet", "type", "propensity_factor", "rain_day_mm", "rain_high_30min_mm", "rain_high_60min_mm", "rain_high_120min_mm", "temp_mean", "temp_max", "temp_min", "wind_mean_kmh", "wind_max_kmh", "cat_covid_group_size_cap", "is_school_holiday", "is_public_holiday", "cat_day_of_week"] |
| ↳↳ `inference`           | `list` | List of expected columns for inference                       | ["location", "type", "propensity_factor", "is_raining", "max_temp", "is_public_holiday", "is_school_holiday", "campaign_name", "campaign_start_date", "campaign_end_date", "campaign_total_costs", "lag_sales"]                                                                                                                                        |

**Data Preprocessing Specifics**

| Parameter                           | Type   | Description                                    | Default Value                                                                                                                                                                                        |
| ----------------------------------- | ------ | ---------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `data_preprocessing`                | `dict` | Default configurations for data preprocessing  | Various                                                                                                                                                                                              |
| ↳ `non_negative_exogeneous_columns` | `list` | Features mandated to have non-negative values  | ["rain_day_mm", "rain_high_30min_mm", "rain_high_60min_mm", "rain_high_120min_mm", "temp_mean", "temp_max", "temp_min", "wind_mean_kmh", "wind_max_kmh", "proxy_revenue", "man_hours", "total_cost"] |
| ↳ `expected_dtypes`                 | `dict` | Expected data types for each feature           | Various                                                                                                                                                                                              |
| ↳ `cost_centre_code`                | `str`  | Expected data type for cost centre code        | "int64"                                                                                                                                                                                              |
| ↳ `man_hours`                       | `str`  | Expected data type for man hours               | "float64"                                                                                                                                                                                            |
| ↳ `proxy_revenue`                   | `str`  | Expected data type for proxy revenue           | "float64"                                                                                                                                                                                            |
| ↳ `location`                        | `str`  | Expected data type for location                | "object"                                                                                                                                                                                             |
| ↳ `outlet`                          | `str`  | Expected data type for outlet                  | "object"                                                                                                                                                                                             |
| ↳ `type`                            | `str`  | Expected data type for type                    | "object"                                                                                                                                                                                             |
| ↳ `propensity_factor`               | `str`  | Expected data type for propensity factor       | "float64"                                                                                                                                                                                            |
| ↳ `rain_day_mm`                     | `str`  | Expected data type for daily rainfall in mm    | "float64"                                                                                                                                                                                            |
| ↳ `temp_mean`                       | `str`  | Expected data type for mean temperature        | "float64"                                                                                                                                                                                            |
| ↳ `wind_mean_kmh`                   | `str`  | Expected data type for mean wind speed in km/h | "float64"                                                                                                                                                                                            |
| ↳ `is_school_holiday`               | `str`  | Expected data type for school holiday flag     | "bool"                                                                                                                                                                                               |
| ↳ `is_public_holiday`               | `str`  | Expected data type for public holiday flag     | "bool"                                                                                                                                                                                               |
| ↳ `campaign_name`                   | `str`  | Expected data type for marketing campaign name | "str"                                                                                                                                                                                                |
| ↳ `date_start`                      | `str`  | Expected data type for campaign start date     | "object"                                                                                                                                                                                             |
| ↳ `total_cost`                      | `str`  | Expected data type for total marketing cost    | "float64"                                                                                                                                                                                            |
| ↳ `cat_day_of_week`                 | `str`  | Expected data type for categorical day of the week | "object"                                                                                                                                                                                             |
| ↳ `date_end`                        | `str`  | Expected data type for the campaign end date       | "object"                                                                                                                                                                                             |
| ↳ `mode`                            | `str`  | Expected data type for the marketing mode          | "str"                                                                                                                                                                                                |


### Key Parameters in the `data_split.yml` File

This sub-section outlines the key parameters in the `data_split.yml` configuration file. It is broken down into sections, each detailing a specific type of data split.

**Simple Split Configuration** 

| Parameter                    | Type  | Description                                            | Default Value  |
| ---------------------------- | ----- | ------------------------------------------------------ | -------------- |
| `training_days`              | `int` | Number of training days                                | 588            |
| `testing_days`               | `int` | Number of testing days                                 | 71             |
| `validation_days`            | `int` | Number of validation days                              | 71             |
| `window_sliding_stride_days` | `int` | Placeholder for config consistency, DO NOT CHANGE THIS | 0              |
| `window_expansion_days`      | `int` | Placeholder for config consistency, DO NOT CHANGE THIS | 0              |
| `split_approach`             | `str` | Splitting approach used                                | "simple_split" |
| `folds`                      | `int` | Number of folds                                        | 1              |

**Sliding Window Configuration**

| Parameter                    | Type  | Description                                            | Default Value    |
| ---------------------------- | ----- | ------------------------------------------------------ | ---------------- |
| `training_days`              | `int` | Number of training days                                | 365              |
| `testing_days`               | `int` | Number of testing days                                 | 60               |
| `validation_days`            | `int` | Number of validation days                              | 60               |
| `window_sliding_stride_days` | `int` | Days to stride the window                              | 90               |
| `window_expansion_days`      | `int` | Placeholder for config consistency, DO NOT CHANGE THIS | 0                |
| `split_approach`             | `str` | Splitting approach used                                | "sliding_window" |
| `folds`                      | `int` | Number of folds                                        | 5                |

**Expanding Window Configuration**

| Parameter                    | Type  | Description                                            | Default Value      |
| ---------------------------- | ----- | ------------------------------------------------------ | ------------------ |
| `training_days`              | `int` | Number of training days                                | 365                |
| `testing_days`               | `int` | Number of testing days                                 | 60                 |
| `validation_days`            | `int` | Number of validation days                              | 60                 |
| `window_sliding_stride_days` | `int` | Placeholder for config consistency, DO NOT CHANGE THIS | 0                  |
| `window_expansion_days`      | `int` | Days set for expanding window for each fold            | 90                 |
| `split_approach`             | `str` | Splitting approach used                                | "expanding_window" |
| `folds`                      | `int` | Number of folds                                        | 5                  |


## How to Execute the Data Pipeline

This guide explains how to operate your data pipeline with the `run_data_pipeline.bat` script. The script offers an automated and convenient way to trigger the pipeline processes.

**Running the Script**

To initiate the pipeline, proceed as follows:

1. **Navigate to the Project Directory**: Ensure you're in the root directory `bipo_demand_forecasting` .

2. **Execute the Script**: 
  
    **Command Line**: Open a Command Prompt in the project directory and execute the following command:

        ```cmd
        # windows powershell
        .\scripts\run_data_pipeline.bat   

        # linux bash
        scripts/run_data_pipeline.bat

        ```

By following these steps, you'll successfully kick off the data pipeline.


### Logging

Logs provide essential insights into the runtime behaviour of your pipeline, enabling effective monitoring the pipeline's execution. 

Below is an example of what you might see in the log output when you run a Kedro pipeline: 

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
18/10/2023 08:24 | kedro | INFO | Column specified for lag generation {'proxyrevenue'}...

```