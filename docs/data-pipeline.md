# Data Pipeline

## Overview
This section outlines the overall architecture, respective modules and the data inputs/outputs that are required or generated in the pipeline. 
Function descriptions are only provided in the sections of [Time-agnostic feature engineering](#5-time-agnostic-feature-engineering) and [Time-dependent feature engineering](#6-time-dependent-feature-engineering) due to module complexities.

The respective modules are largely dependent on `pipeline.py` (comprises of functions from `nodes.py`) and `nodes.py` containing functions located in `src/bipo/pipelines` folder.

## Data Sources
| Filename | Data type | Description | Data Source |
| --- | --- | --- | --- |
| `proxy_revenue_masked.csv` | csv | Daily proxy revenue of all outlets in csv format. | BIPO                                            |
| `marketing cost.xlsx` | xlsx | Cost breakdown of marketing campaigns by mode. | BIPO  |
| `consumer propensity to spend.xlsx` | xlsx | Daily consumer propensity to spend by regions in Singapore. | BIPO |
| `covid capacity.csv` | csv | Extracted sheet containing group size limits implemented as part of COVID pandemic restriction (Sheet 2 of provided covid capacity.xlsx) | BIPO        
| `SG climate records 2021 - 2022.xlsx` | xlsx | Daily climate data from four key regions in Singapore. | BIPO |
| `holiday_df.xlsx` | xlsx | Mapping of past dates to school holidays and public holidays in Singapore. | MOE Website, School Terms and Holidays for 2022 |
---

## Architecture

The data pipeline architecture diagram below provides a high-level overview of the key processing steps in transforming raw data for model training.

![Pipeline Design](./assets/data_pipeline_training.png)

| Submodules | Activity | Configuration File^ | Output Store  |
| --- | --- | --- | --- |
| Raw Data | Initial datasets provided/crafted in either `.csv` and `.xlsx` files. For a detailed list of these sources, refer to the [Data Sources section](#data-sources) above. | - | - |
| Data Loader | Ingests raw data files and restructures data into daily based records for datasets containing multiple same-date entries for different features categories, while datafiles containing unique daily records are merged on into a single dataset. | `constants.yml` | Loaded restructured data |
| Data Preprocessing | Combines all non-proxy revenue features with respective outlet proxy revenue files while filtering out outlets based on specified conditions. | `constants.yml`, `parameters.yml` | Preprocessed Data |
| Data Merge | Combines individual processed outlet files into single file to facilitate time-based split handled in the next module. | - | Merged Outlet Data |
| Data Split | Implements following configurable time-based data split as follows: <ul><li>Simple split;</li><li>Expanding window; and</li><li> Sliding_window.</li></ul> into following sets: <ul><li>Training;</li><li>Validation; and </li><li>Testing.</li></ul>| `constants.yml`, `parameters.yml`, `conf/base/data_split.yml` | Split Data              |
| Time Agnostic Feature Engineering  | Feature engineering processes that are not time dependent. This includes:<ul><li>Boolean feature creation based on defined condition(s);</li><li>Adstock feature engineering;</li><li>Differencing of (multiple) paired features; and</li><li>General value imputation.</li></ul> | `parameters.yml` | Feature Engineered Data |
| Time Dependent Feature Engineering |Feature engineering processes that are time dependent. This includes:<ul><li>One-hot/Ordinal encoding;</li><li>Standardisation/Normalisation of values; and</li><li>*tsfresh* feature engineering.</li></ul> | `constants.yml`, `parameters.yml` | Feature Engineered Data |
| Model-specific Preprocessing | Processes data based on specific requirements of the model (if necessary). Subsequently, conducts removal of data points containing null feature values or single valued columns.| `parameters.yml` | Model Input Data |
---

> Note: `constants.yml` is used as a fallback when invalid values are set in `parameters.yml` OR assumed defaults for processing. Not all parameters are covered as applying default values do not make sense. 

### Design Considerations
1. **Modularity**: Flexibility and maintainability of components.
2. **Module Decoupling**: Faster adaptation to changing requirements without significant overall impact to entire program as much as possible.
3. **Configurability**: To facilitate experimentation of various data/model manipulataion approaches.

## 1. Data Loader
This submodule focuses on ingesting and restructuring raw and multiple common time-indexed data into unique daily-based time-index representations, while unique daily records are merged based on common time-index. Files are then segregate into outlet proxy revenue and non revenue categories.

The diagram below provides a general overview of the submodule. 

![Pipeline Design](./assets/data_loader.png)

### Input(s)
| Component | Description | Data directory |
| --- | --- | --- |
| Multi daily records (csv/) | Folder containing csv files representing mutliple common time-indexed to features mappings. | data/01_raw/multi_daily_records/csv/
| Multi daily records (xlsx/) | Folder containing xlsx files representing mutliple common time-indexed to features mappings. | data/01_raw/multi_daily_records/xlsx/
| Unique daily records (csv/) | Folder containing csv files representing unique time-indexed to features mappings. | data/01_raw/unique_daily_records/csv/
| Unique daily records (xlsx/) | Folder containing xlsx files representing unique time-indexed to features mappings. | data/01_raw/unique_daily_records/xlsx/
---

### Output(s)
| Component | Description | Data directory |
| --- | --- | --- |
| Non proxy revenue files | Folder containing processed time-indexed files representing various data sources in .csv which are non-proxy revenue. | data/02_dataloader/non_revenue_partitions/
| Outlet proxy revenue files | Folder containing individual time-indexed outlet-based proxy revenue in .csv format. | data/02_dataloader/outlet_proxy_revenues/
---

### Important Note on Pipeline Execution
For pipeline execution with Kedro, Data Loader submodule needs to be executed first separately before executing other modules that follows subsequently.
 
This is due to Data Loader submodule output definition which explicitly defines a mapping for processed file for each file input, whereas the input to Data Preprocessing submodule is folder-specific definition (where Data Loader processed file(s) resides) is seen by Kedro to be different entities. As a result, Data Preprocessing maybe executed when not all files are fully processed in the Data Loader submodule.

## 2. Data Preprocessing
The data_preprocessing submodule applies feature combining using all files non-proxy revenue datasets with individual outlet proxy revenue datasets using generated outputs in previous submodule, [Data Loader](#1-data-loader). Data filterings comprising the following are implemented:
- Outlets containing data points within required dates are extracted.
- Specified outlets to be excluded.
- Outlets with zero-values exceeding a proportion are filtered out (configurable in parameters.yml). This is to prevent binning issues downstream prior to modeling as equal frequency binning is used. 

The diagram below provides a general overview of the submodule. 

![Pipeline Design](./assets/data_preprocessing.png)

### Input(s)
| Component | Description | Data directory |
| --- | --- | --- |
| Non proxy revenue files | Folder containing processed time-indexed files representing various data sources in .csv format which are non-proxy revenue related. | data/02_dataloader/non_revenue_partitions/
| Outlet proxy revenue files | Folder containing individual time-indexed outlet-based proxy revenue in .csv format. | data/02_dataloader/outlet_proxy_revenues/
---

### Output(s)
| Component | Description | Data directory |
| --- | --- | --- |
| Processed outlet datafiles | Folder containing individual time-indexed outlet-based proxy revenue with merged features in .csv format. | data/03_data_preprocessing/processed/ |
---

## 3. Data Merge
This submodule concatenates the individual outlet proxy revenue dataset files generated in previous submodule, [Data Preprocessing](#2-data-preprocessing) into a single file to facilitate time-index split which is handled in next submodule, [Data Split](#4-data-split).

The diagram below provides a general overview of the submodule.

![Pipeline Design](./assets/data_merge.png)

### Input(s)
| Component | Description | Data directory |
| --- | --- | --- |
| Processed outlet datafiles | Folder containing individual time-indexed outlet-based proxy revenue with merged features in .csv format. | data/03_data_preprocessing/processed/ | This data is then passed through different data splitting approaches like simple split, expanding window , and sliding window, as specified in the configuration file. |
---

### Output(s)

| Component | Description | Data directory |
| --- | --- | --- |
| Merged outlet datafiles | File containing merged time-indexed outlet-based proxy revenue with merged features in .csv format. | data/04_data_split/data_merged/data_merged.csv |
---

## 4. Data Split

This submodule takes the output of [Data Merge](#3-data-merge) and implements all **three** time-dependent data splits, namely 
- simple split;
- expanding window; and
- sliding window.

Split parameters are configurable via `conf/base/parameters/data_split.yml`. Before conducting splits, checks would be made to validate inputs. If invalid inputs are set, `conf/base/constants.yml` containing default fallback values would be used.

The diagram below provides a general overview of the submodule.

![Pipeline Design](./assets/data_split.png)

### Input(s)
| Component | Description | Data directory |
| --- | --- | --- |
| Merged outlet datafile | Single file containing all outlet-based proxy revenue in .csv format. | data/04_data_split/data_merged/data_merged.csv |
---

### Output(s)
| Component | Description | Data directory |
| --- | --- | --- |
| Simple split | Folder containing training, validation and testing datasets generated via simple time-based index split configured by days. | data/04_data_split/simple_split/ |
| Expanding window split | Folder containing training, validation and testing datasets generated using expanding window time-index split configured by days. | data/04_data_split/expanding_window/ |
| Sliding window split | Folder containing training, validation and testing datasets generated using sliding window time-index split configured by days. | data/04_data_split/sliding_window/ |
---

## 5. Time-Agnostic Feature Engineering
This submodule serves to facilitate feature engineering processes which are  time-agnostic based on configured data split source using fold information and name of data split as configured in `parameters.yml`. Upon completion of necessary feature engineering works, each training, validation and test folds are partitioned into files by outlets and stored in respective `training`, `validation` and `test` folders. 

The key feature engineering works in this submodule are as follows:
- Boolean indicator feature generation based on conditions (under `feature_indicator_diff_creation.py`). 
    - Example: *`is_raining`* feature generated from *`daily_rainfall_total_mm`* feature based whether value is greater than > 0.2 condition.
- Differencing of values using list of 2 columns features (under `feature_indicator_diff_creation.py`)
- Marketing cost imputation for days without any marketing events

The diagram below provides a general overview of the submodule.

![Pipeline Design](./assets/time_agnostic_feature_engineering.png)

### Core Functions

| Function name | Feature of interest (snakecased feature names) | Description | New feature name | 
| --- | --- | --- | --- |
|`create_min_max_feature_diff` | Based on `columns_to_diff_list` parameter (expects List of list containing 2 elements within) | Calculates the difference in values between 2 columns | `diff_<min_column_name>_<max_column_name>`|
|`create_is_weekday_feature` | Dataframe index in datetime | Sets to 1 if the derived weekday information value <5, else 0.| `is_weekday`|
|`create_is_holiday_feature` | Based on `fe_holiday_column_list` parameter (list). Example features used: `school_holiday`, `public_holiday` | Sets to 1 if entry exists, else 0.| Prefix `is_` added to all columns used. Example: `is_school_holiday`, `is_public_holiday`|
|`create_is_raining_feature` | Based on `fe_rainfall_column` parameter (string). Example feature used: `daily_rainfall_total_mm` | Set to 1 if column reference is greater than 0.2, else 0. | Appends a prefix 'is_' to column used. Example: `is_daily_rainfall_total_mm`|
|`create_is_pandemic_feature` | Based on `fe_pandemic_column` parameter (string). Example feature used: `group_size_cap` | Set to 0 if column reference indicates "no limit", else 1 | `is_pandemic_restrictions` |
|`create_mkt_campaign_counts_start_end`||||
|`generate_adstock` | Based on `mkt_channel_list` parameter (list). Example feature used: `tv_ad_daily_cost` | Calculates the adstock values for each marketing channel daily cost. | Prefix `adstock_` is added to all columns used. Example: `adstock_tv_ad_daily_cost` |
|`no_mkt_days_imputation`| Based on `mkt_columns_to_impute_dict` parameter (dict) | Utilises `mkt_columns_to_impute_dict` containing imputation values for specified marketing cost features for days without any marketing events.| No new features created. Only affects `tv_ad_daily_cost`, `radio_ad_daily_cost`, `instagram_ad_daily_cost`, `facebook_ad_daily_cost`, `youtube_ad_daily_cost`,`poster_campaign_daily_cost`, `digital_daily_cost` |
---

### Input(s)

| Component | Description | Data Directory |
| --- | --- | --- |
| Simple split | Folder containing training, validation and testing datasets folds generated using simple time-index split configured by days. | data/04_data_split/simple_split/ |
| Expanding window split | Folder containing training, validation and testing datasets folds generated using expanding window time-index split configured by days. | data/04_data_split/expanding_window/ |
| Sliding window split | Folder containing training, validation and testing datasets folds generated using sliding window time-index split configured by days. | data/04_data_split/sliding_window/ |
---

### Output(s)

| Component | Description | Data Directory |
| --- | --- | --- |
| Training dataset | Training dataset folder containing fold-outlet files with engineered columns. | data/04a_time_agnostic_feature_engineering/sliding_window/training/ |
| Validation dataset | Validation dataset folder containing fold-outlet files with engineered columns. | data/04a_time_agnostic_feature_engineering/sliding_window/validation/ |
| Testing dataset | Testing dataset folder  containing fold-outlet files with engineered columns. | data/04a_time_agnostic_feature_engineering/sliding_window/testing/ |
---

## 6. Time-Dependent Feature Engineering
This submodule serves to facilitate feature engineering processes which are supposedly time-dependent (or time-sensitive) based on outputs generated under the previous section.

The key feature engineering works in this submodule are as follows:
- One-hot and ordinal encodings
- StandardScaler and Normalizer
- Lag feature generation
- *tsfresh* feature generation **(can be enabled/disabled via parameters.yml)**
- Merge all feature engineered columns together (including *tsfresh*)

The diagram below provides a general overview of the submodule.

![Pipeline Design](./assets/time_dependent_feature_engineering.png)

### Core Functions
| Function name | Feature of interest (snakecased feature names) | Description | New feature name | 
| --- | --- | --- | --- |
| `apply_binning_fit` | Based on `binning_dict` parameter (dict) containing feature to bin as key with bin labels in a list specified in parameters.yml.|Function applies a 'fit' to binning to learn binned parameters. | - |
| `apply_binning_transform` | Based on `binning_dict` parameter (dict) containing column to bin as key with bin labels in a list specified in parameters.yml. | Function applies a 'transform' for binning using binned parameters.| Column to be would be prefixed with `binned_`. Example: `binned_proxyrevenue` |
| `apply_standard_norm_fit` | Based on `columns_to_std_norm_dict` parameter (list) containing feature to be standardised or normalized specified in parameters.yml.|Applies 'fit' method with either sklearn standardscaler/normalizer depending on `normalization_approach` parameter option which can be either `normalize` or `standardize` with default `normalize` used for invalid cases. | - |
| `apply_standard_norm_transform` | Based on `columns_to_std_norm_dict` parameter (list) containing feature to be standardised or normalized specified in parameters.yml.|Applies 'transform' method with either sklearn standardscaler/normalizer depending on `normalization_approach` parameter option which can be either `normalize` or `standardize` with default `normalize` used for invalid cases. | No new columns generated as values used in `columns_to_std_norm_dict` are overwritten. |
| `generate_lag` | Based on `columns_to_create_lag_features` parameter (list) indicating column for lag features generation specified in parameters.yml. | Generates simple lag, simple moving average with period shift, and weekly average lags. Configured lag parameters with `lag_periods_list`, `sma_windows_periods_list`,  `sma_tsfresh_shift_period`, `lag_week_periods_list` | New features identified with `lag_` prefix constructed with corresponding duration and the column used as suffix. Example:  <ul><li>`lag_9_proxyrevenue`,`lag_14_proxyrevenue` (simple lag)</li><li>`lag_9_sma_7_days_proxyrevenue` (simple moving average)</li><li>`lag_mean_1_week_proxyrevenue`, `lag_mean_2_week_proxyrevenue` (lag weekly )</li></ul> 
| `apply_feature_encoding_fit` | Based on `fe_ordinal_encoding_dict` parameter (dict) and `fe_one_hot_encoding_col_list` parameter (list) specified in parameters.yml.|Applies fit on columns identified for one-hot encoding or ordinal encoding using sklearn library. | - |
| `apply_feature_encoding_transform` | Based on learned encodings from generated artefacts from `apply_feature_encoding_fit` function specified in parameters.yml. |Applies transform on columns identified for one-hot encoding or ordinal encoding using sklearn library based on learned encodings. | For one-hot encoding, learned encoding names are used. For ordinal encoding, a prefix of `ord_` is appended to columns used. |
---

### Generated Artefacts
| Name | File Type | Directory Path |
| --- | --- | --- |
| `feature_encoding.pkl` | pickle | data/05_feature_engineering/artefacts/feature_encoding.pkl |
| `std_norm.pkl` | pickle | data/05_feature_engineering/artefacts/std_norm.pkl |
| `fold<number>_tsfresh_relevant_features.json` | json | data/05_feature_engineering/tsfresh_features/artefacts/tsfresh_relevant_features/|
---

### *tsfresh* (configurable for enabling/disabling)
The following functions **are skipped** if *`include_tsfresh`* parameter in `parameters.yml` is set to `False`.
| Function name | Feature of interest (snakecased feature names) | Description | New feature name | 
| --- | --- | --- | --- |
| `run_tsfresh_feature_selection_process` | Based on `fe_target_feature_name` (numeric target feature) parameter specified in parameters.yml. | Derives and construct relevant tsfresh features based on dataframe predictors and predicted features. |-|
| `run_tsfresh_feature_engineering_process` |Based on `fe_target_feature_name` (numeric target feature) parameter specified in parameters.yml. | Creates tsfresh features based on learnt tsfresh artefacts | New feature names are prefixed by value configured in `fe_target_feature_name`. Example with 'proxyrevenue' used `proxyrevenue_cwt_coefficients_coeff_0_w_5_widths_2_5_10_20` |
---

### Input(s)

| Component | Description | Data directory |
| --- | --- | --- |
| Training dataset | Training dataset folder containing fold-outlet files with engineered columns. | data/04a_time_agnostic_feature_engineering/sliding_window/training/ |
| Validation dataset | Validation dataset folder containing fold-outlet files with engineered columns. | data/04a_time_agnostic_feature_engineering/sliding_window/validation/ |
| Testing dataset | Testing dataset folder  containing fold-outlet files with engineered columns. | data/04a_time_agnostic_feature_engineering/sliding_window/testing/ |
---

### Intermediate Files

| Component | Description | Data directory |
| --- | --- | --- |
| Training dataset (processed) | Training dataset folder containing engineered features **without *tsfresh***. | data/05_feature_engineering/no_tsfresh/training/ |
| Validation dataset (processed) | Validation dataset folder containing engineered features **without *tsfresh***. | data/05_feature_engineering/no_tsfresh/validation/ |
| Testing dataset (processed) | Validation dataset folder containing engineered features **without *tsfresh***. | data/05_feature_engineering/no_tsfresh/testing/ |
| Lag features dataset | Folder containing generated lag features for each outlet (using entire dataset). | data/05_feature_engineering/lag_features/ |
| *tsfresh* features dataset | Folder containing generated derived tsfeatures for each outlet per fold. | data/05_feature_engineering/tsfresh/ |

### Output(s)

| Component | Description | Data directory |
| --- | --- | --- |
| Training dataset | Training dataset folder containing engineered features with *tsfresh* features | data/05_feature_engineering/features_merged/training/ |
| Validation dataset | Validation dataset folder containing engineered features with *tsfresh* features| data/05_feature_engineering/features_merged/validation/ |
| Testing dataset | Testing dataset folder containing engineered features with *tsfresh* features| data/05_feature_engineering/features_merged/testing/ |
---

## 7. Model-specific Preprocessing

This submodule is primarily used to conduct any model-specific preprocessing steps, especially when more models are experimented with some requiring specialised inputs conversion, continuing from [Time-dependent feature engineering](#6-time-dependent-feature-engineering) submodule.

Subsequently, numerical columns are retained with the removal of non-numerical columns, redundant columns (i.e. single-valued column(s)) and rows containing null values are implemented before splitting the dataset into predictors and predicted feature for model training purposes. This is to ensure model training can proceed without issues.

> Note: The existing OrderedModel and EBM used do not have a specific preprocessing step required that differs from each other. Should there be new models be considered, this submodule is to be utilised for necessary model-specific preprocessing works.

The diagram below provides a general overview of the submodule.

![Pipeline Design](./assets/model_specific_preprocessing.png)

### Input(s)
Feature-engineered data customised for the specific predictive models being used in the analysis.

| Component | Description | Data directory |
| --- | --- | --- |
| Training dataset | Training dataset folder containing engineered features with *tsfresh* features | data/05_feature_engineering/features_merged/training/ |
| Validation dataset | Validation dataset folder containing engineered features with *tsfresh* features| data/05_feature_engineering/features_merged/validation/ |
| Testing dataset | Testing dataset folder containing engineered features with *tsfresh* features| data/05_feature_engineering/features_merged/testing/ |
---

### Output(s)

| Component | Description | Data directory |
| --- | --- | --- |
| Training dataset | Training dataset folder containing processed features (removed null rows/single-constant columns) with all engineered features. | data/06_model_specific_preprocessing/<ordered_model or ebm>/training/ |
| Validation dataset | Validation dataset folder containing processed features (removed null rows/single-constant columns) with all engineered features.| data/06_model_specific_preprocessing/<ordered_model or ebm>/validation/ |
| Testing dataset | Testing dataset folder containing processed features (removed null rows/single-constant columns) with all engineered features.| data/06_model_specific_preprocessing/<ordered_model or ebm>/testing/ |
---