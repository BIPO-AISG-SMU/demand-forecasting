# Archive of Model Experiments

## Datasets 

This section details the information datasets used for model experiments from Sprint 8 to Release Sprint, which includes the dataset versions, list of features and the data split types used for training and validation of models.

### Sprint 8 (Dataset Version 6)

The dataset integrates core features with additional features derived from the [LightweightMMM library](https://lightweight-mmm.readthedocs.io/en/latest/index.html). The features are listed in the table below.

|Core Features                           | LightweightMMM Features                    |
|:---------------------------------------|--------------------------------------------|
| binned_proxy_revenue                   | adstock_channel_daily_cost                 |
| campaign_daily_cost                    | carryover_channel_daily_cost               |
| cat_mkt_campaign_end                   |                                            |
| cat_mkt_campaign_start                 |                                            |
| count_mkt_campaign                     |                                            |
| day_of_week                            |                                            |
| digital_daily_cost                     |                                            |
| facebook_ad_daily_cost                 |                                            |
| instagram_ad_daily_cost                |                                            |
| is_having_campaign                     |                                            |
| is_public_holiday                      |                                            |
| is_raining                             |                                            |
| is_school_holiday                      |                                            |
| is_weekend                             |                                            |
| lag_14_days_proxy_revenue              |                                            |
| lag_1_week_mean_weekly_proxy_revenue   |                                            |
| lag_2_week_mean_weekly_proxy_revenue   |                                            |
| lag_9_days_proxy_revenue               |                                            |
| pandemic_restrictions                  |                                            |
| poster_campaign_daily_cost             |                                            |
| radio_ad_daily_cost                    |                                            |
| sma_window_7_days_proxy_revenue        |                                            |
| sma_window_8_days_proxy_revenue        |                                            |
| tv_ad_daily_cost                       |                                            |
| youtube_ad_daily_cost                  |                                            |

#### Dataset Splits and Structures

Different temporal splits involving simple split, sliding window and expanding window were applied to the dataset as part of model performance evauation. Applying splits would reduce likelihood of model overfitting during training, thus making it generalisable for future unseen data.

| Split Type      | Version | Training Days | Validation Days | Testing Days | Train Set Count | Validation Set Count | Total Count | Feature Count | Remarks |
|-----------------|---------|---------------|-----------------|--------------|-----------------|----------------------|-------------|--------------|---------|
| Simple Split    | 6.1    | 588*          | 71              | 71           | 32,604          | 4,047                | 36,651      | 31           | Data spans across two full years for a comprehensive temporal analysis. |
| Sliding Window  | 6.2    | 365* (per fold) | 30 (per fold) | 30 (per fold) | Fold 1: 19,893<br/>Fold 2: 20,805<br/>Fold 3: 20,805<br/>Fold 4: 20,805 | Fold 1: 1,710<br/>Fold 2: 1,710<br/>Fold 3: 1,710<br/>Fold 4: 1,710 | Fold 1: 21,603<br/>Fold 2: 22,515<br/>Fold 3: 22,515<br/>Fold 4: 22,515 | 31, 29 after feature removal | Sequential data windows are used to emulate a rolling forecast scenario. |
| Sliding Window  | 6.3    | 365* (per fold) | 60 (per fold) | 60 (per fold) | Fold 1: 19,893<br/>Fold 2: 20,805<br/>Fold 3: 20,805 | Fold 1: 3,420<br/>Fold 2: 3,420<br/>Fold 3: 3,420 | Fold 1: 23,313<br/>Fold 2: 24,225<br/>Fold 3: 24,225 | 31, 29 after feature removal | Each window incrementally shifts forward to test model adaptability over time. |
| Expanding Window | 6.4    | 349* to 619 (expanding) | 30 (per fold) | 30 (per fold) | Fold 1: 19,893<br/>Fold 2: 25,023<br/>Fold 3: 30,153<br/>Fold 4: 35,283 | Fold 1: 1,710<br/>Fold 2: 1,710<br/>Fold 3: 1,710<br/>Fold 4: 1,710 | Fold 1: 21,603<br/>Fold 2: 26,733<br/>Fold 3: 31,863<br/>Fold 4: 36,993 | 31 | Training set progressively includes more data to reflect a growing sample size. |
| Expanding Window | 6.5    | 349* to 529 (expanding) | 60 (per fold) | 60 (per fold) | Fold 1: 19,893<br/>Fold 2: 25,023<br/>Fold 3: 30,153 | Fold 1: 3,420<br/>Fold 2: 3,420<br/>Fold 3: 3,420 | Fold 1: 23,313<br/>Fold 2: 28,443<br/>Fold 3: 33,573 | 31 | Expanding training period allows for assessment of model performance over extended time frames. |

\* Total training days/dataset points reduced with the inclusion of lag features after feature engineering.


### Sprint 9 (Dataset Version 7)

In Sprint 9, new experiments were performed on the updated dataset, with refined feature sets and split strategies. This dataset consists of the core features and additional marketing and [tsfresh](https://tsfresh.readthedocs.io/en/latest/) features.

| Core Features                 | *LightweightMMM*  Features                 | *tsfresh* Features                                                  |
|-------------------------------|--------------------------------------------|--------------------------------------------------------------------|
| is_daily_rainfall_total_mm    | adstock_digital_daily_cost                 | proxyrevenue_agg_linear_trend_attr_intercept_chunk_len_5_f_agg_mean|
| is_name_end                   | adstock_facebook_ad_daily_cost             | proxyrevenue_cwt_coefficients_coeff_1_w_10_widths_2_5_10_20        |
| is_name_start                 | adstock_instagram_ad_daily_cost            | proxyrevenue_cwt_coefficients_coeff_1_w_20_widths_2_5_10_20        |
| is_public_holiday             | adstock_poster_campaign_daily_cost         | proxyrevenue_cwt_coefficients_coeff_2_w_5_widths_2_5_10_20         |
| is_school_holiday             | adstock_radio_ad_daily_cost                |                                                                    |
| is_weekday                    | adstock_tv_ad_daily_cost                   |                                                                    |
| lag_14_proxyrevenue           | adstock_youtube_ad_daily_cost              |                                                                    |
| lag_9_proxyrevenue            | carryover_digital_daily_cost               |                                                                    |
| lag_9_sma_7_days_proxyrevenue | carryover_facebook_ad_daily_cost           |                                                                    |
| lag_mean_1_week_proxyrevenue  | carryover_instagram_ad_daily_cost          |                                                                    |
| lag_mean_2_week_proxyrevenue  | carryover_poster_campaign_daily_cost       |                                                                    |
| type_dine-in                  | carryover_radio_ad_daily_cost              |                                                                    |

#### Data Splits

The data split for Sprint 9 is as follows:

| Split Type      | Version | Training Days            | Validation Days | Train Set Count                                            | Validation Set Count                                      | Total Count                                               | Remarks                                                                                       |
|-----------------|---------|--------------------------|-----------------|------------------------------------------------------------|-----------------------------------------------------------|------------------------------------------------------------|----------------------------------------------------------------------------------------------|
| Expanding Window | 7.4    | 349* to 619              | 30              | Fold 1: 19,893<br/>Fold 2: 25,023<br/>Fold 3: 30,142<br/>Fold 4: 35,272 | Fold 1: 855<br/>Fold 2: 855<br/>Fold 3: 855<br/>Fold 4: 855   | Fold 1: 20,748<br/>Fold 2: 25,878<br/>Fold 3: 30,997<br/>Fold 4: 36,127 | Each fold increases in size to simulate accumulating data over time, with adjustments for advanced feature engineering. |

\* Indicates a reduction in training and validation days/dataset points due to the inclusion of lag features after *tsfresh* feature inclusion.

### Release Sprint (Dataset Version 8)

For Release Sprint, the dataset has been curated to include a focused set of core features alongside lag features and adstock features. Notably, the generation of adstock features no longer relies on the LightweightMMM library.

| Core Features               | Lag Features + Encoded Feature | Adstock Features                |
|-----------------------------|--------------------------------|---------------------------------|
| maximum_temperature_c       | lag_9_proxyrevenue             | adstock_tv_ad_daily_cost        |
| minimum_temperature_c       | lag_14_proxyrevenue            | adstock_radio_ad_daily_cost     |
| factor                      | lag_9_sma_7_days_proxyrevenue  | adstock_instagram_ad_daily_cost |
| tv_ad_daily_cost            | lag_mean_1_week_proxyrevenue   | adstock_facebook_ad_daily_cost  |
| radio_ad_daily_cost         | lag_mean_2_week_proxyrevenue   | adstock_youtube_ad_daily_cost   |
| instagram_ad_daily_cost     | type_dine_in                   | adstock_poster_campaign_daily_cost |
| facebook_ad_daily_cost      |                                | adstock_digital_daily_cost      |
| youtube_ad_daily_cost       |                                |                                 |
| poster_campaign_daily_cost  |                                |                                 |
| digital_daily_cost          |                                |                                 |
| name_counts                 |                                |                                 |
| is_name_start               |                                |                                 |
| is_name_end                 |                                |                                 |
| is_weekday                  |                                |                                 |
| is_school_holiday           |                                |                                 |
| is_public_holiday           |                                |                                 |
| is_daily_rainfall_total_mm  |                                |                                 |

#### Dataset Splits

In Release Sprint, different types of splits were generated to evaluate model performance over various periods.

| Split Type       | Version | Training Days | Validation Days | Testing Days | Train Set Count                                | Validation Set Count | Total Count                                     | Remarks                                                 |
|------------------|---------|---------------|-----------------|--------------|------------------------------------------------|----------------------|--------------------------------------------------|---------------------------------------------------------|
| Simple Split     | 8.1    | 588*          | 71              | 71           | 32,604                                         | 4,047                | 40,698                                           | With adstock features, without tsfresh                 |
| Sliding Window   | 8.2    | 365*          | 60              | 60           | Fold 1: 19,893<br/>Fold 2: 20,805<br/>Fold 3: 20,805 | 3,420 per fold       | Fold 1: 26,733<br/>Fold 2: 27,645<br/>Fold 3: 27,645 | Similar to 8.1, then tailored using Sliding Window Split |
| Expanding Window | 8.3    | 365*          | 60              | 60           | Fold 1: 19,893<br/>Fold 2: 25,023<br/>Fold 3: 30,153 | 3,420 per fold       | Fold 1: 26,733<br/>Fold 2: 31,863<br/>Fold 3: 36,993 | Similar to 8.1, then tailored using Expanding Window Split |
| Simple Split     | 8.4    | 588*          | 71              | 71           | 32,604                                         | 4,047                | 40,698                                           | Without adstock and tsfresh                           |
| Sliding Window   | 8.5    | 365*          | 60              | 60           | Fold 1: 19,893<br/>Fold 2: 20,805<br/>Fold 3: 20,805 | 3,420 per fold       | Fold 1: 26,733<br/>Fold 2: 27,645<br/>Fold 3: 27,645 | Similar to 8.4, then tailored using Sliding Window Split |
| Expanding Window | 8.6    | 365*          | 60              | 60           | Fold 1: 19,893<br/>Fold 2: 25,023<br/>Fold 3: 30,153 | 3,420 per fold       | Fold 1: 26,733<br/>Fold 2: 31,863<br/>Fold 3: 36,993 | Similar to 8.4, then tailored using Expanding Window Split |
| Simple Split     | 8.7    | 588*          | 71              | 71           | 32,604                                         | 3,135                | 38,874                                           | With adstock and a limited set of tsfresh features     |
| Sliding Window   | 8.8    | 365*          | 60              | 60           | 19,893                                         | 2,565                | 25,023                                           | Similar to v8.7, then tailored using Sliding Window Split |
| Expanding Window | 8.9    | 365*          | 60              | 60           | Fold 1: 19,893<br/>Fold 2: 25,023<br/>Fold 3: 30,142 | 2,565 per fold       | Fold 1: 25,023<br/>Fold 2: 30,153<br/>Fold 3: 35,272 | Similar to v8.7, then tailored using Expanding Window Split |

\*Training, validation and test days adjusted to account for feature engineering, such as the inclusion of lag features.


## Model Experiments

In this section, we compile the results of our model experiments conducted between Sprint 8 and Release Sprint, which were aimed to assess the performance of our models based on the metrics of precision,recall and accuracy under various conditions and datasets.

### Experiment 1: Data Split Comparison

This series of experiments were conducted to evaluate the performance of models across different data split approaches, namely: simple split, sliding window split, and expanding window split. The following default parameters for each model were used:
- **EBM**: outer_bags = 8, inner_bags = 0, learning_rate = 0.01, max_leaves = 3, min_samples_leaf = 2
- **OrderedModel**: distr = "probit," method = "bfgs," max_iter = 50

#### Sprint 8 Results

The results of Sprint 8 model experiments performed on [datasets version 6](#sprint-8-dataset-version-6) are summarised in the table below.

| Dataset Version  | Data Split                  | Model   | Validation Precision (Average per fold) | Validation Recall (Average per fold) | Validation Accuracy (Average per fold) | Train Precision (Average per fold) | Train Recall (Average per fold) | Train Accuracy (Average per fold) |
|-----------|-----------------------------|---------|---------------------------------------|------------------------------------|--------------------------------------|----------------------------------------|-------------------------------------|---------------------------------------|
| 6.1      | Simple Split                | EBM     | 0.411                                | 0.420                             | 0.420                               | 0.502                                  | 0.511                              | 0.511                                |
|           |                             | Ordered | 0.458                                | 0.450                             | 0.450                               | 0.584                                  | 0.592                              | 0.592                                |
| 6.2 (30days) | Sliding Window (4 folds)  | EBM     | 0.454                                | 0.431                             | 0.431                               | 0.571                                  | 0.580                              | 0.580                                |
|               |                          | Ordered | 0.468                                | 0.471                             | 0.471                               | 0.502                                  | 0.512                              | 0.512                                |
| 6.3 (60days) | Sliding Window (4 folds)  | EBM     | 0.458                                | 0.446                             | 0.446                               | 0.577                                  | 0.585                              | 0.585                                |
|               |                          | Ordered | 0.463                                | 0.464                             | 0.464                               | 0.502                                  | 0.509                              | 0.509                                |
| 6.4 (30days) | Expanding Window (4 folds)| EBM     | 0.472                                | 0.426                             | 0.426                               | 0.584                                  | 0.592                              | 0.592                                |
|               |                          | Ordered | 0.464                                | 0.480                             | 0.480                               | 0.508                                  | 0.516                              | 0.516                                |
| 6.5 (60days) | Expanding Window (3 folds)| EBM     | 0.477                                | 0.411                             | 0.411                               | 0.584                                  | 0.591                              | 0.591                                |
|               |                          | Ordered | 0.453                                | 0.463                             | 0.452                               | 0.512                                  | 0.520                              | 0.520                                |

#### Sprint 9 Results

The outcomes of Sprint 9 experiments on [dataset version 7](#sprint-9-dataset-version-7) are summarised in the table below.

| Fold | Model   | Validation Precision | Validation Recall | Validation Accuracy | Validation Period      | Train Precision | Train Recall | Train Accuracy |
|------|---------|---------------------------|------------------------|--------------------------|------------------------|----------------------|-------------------|---------------------|
| 1    | EBM     | 0.522                     | 0.493                  | 0.493                    | 16/1/22 to 30/1/22     | 0.584                | 0.591             | 0.591               |
|      | Ordered | 0.507                     | 0.444                  | 0.444                    |                        | 0.529                | 0.536             | 0.536               |
| 2    | EBM     | 0.475                     | 0.481                  | 0.481                    | 16/4/22 to 30/4/22     | 0.570                | 0.577             | 0.577               |
|      | Ordered | 0.522                     | 0.545                  | 0.545                    |                        | 0.517                | 0.525             | 0.525               |
| 3    | EBM     | 0.497                     | 0.418                  | 0.418                    | 15/7/22 to 29/7/22     | 0.565                | 0.574             | 0.574               |
|      | Ordered | 0.051                     | 0.226                  | 0.226                    |                        | 0.06                 | 0.251             | 0.251               |
| 4    | EBM     | 0.470                     | 0.478                  | 0.478                    | 13/10/22 to 27/10/22   | 0.559                | 0.568             | 0.568               |
|      | Ordered | 0.097                     | 0.313                  | 0.313                    |                        | 0.064                | 0.253             | 0.253               |

#### Release Sprint Results

| Dataset Version  | Data Split                  | Model   | Validation Precision (Average per fold) | Validation Recall (Average per fold) | Validation Accuracy (Average per fold) | Test Precision (Average per fold) | Test Recall (Average per fold) | Test Accuracy (Average per fold) |
|-----------|-----------------------------|---------|---------------------------------------|------------------------------------|--------------------------------------|---------------------------------------|------------------------------------|--------------------------------------|
| 8.1      | Simple Split                | EBM     | 0.427                                | 0.440                             | 0.440                               | 0.483                                 | 0.479                             | 0.479                                |
|           |                             | Ordered | 0.414                                | 0.418                             | 0.418                               | 0.465                                 | 0.425                             | 0.425                                |
| 8.2      | Sliding Window              | EBM     | 0.491                                | 0.445                             | 0.445                               | 0.469                                 | 0.452                             | 0.452                                |
|           |                             | Ordered | 0.485                                | 0.463                             | 0.463                               | 0.418                                 | 0.321                             | 0.321                                |
| 8.2      | Sliding Window (Fold 2)     | EBM     | 0.473                                | 0.485                             | 0.485                               | 0.453                                 | 0.438                             | 0.438                                |
|           |                             | Ordered | 0.473                                | 0.492                             | 0.492                               | 0.463                                 | 0.471                             | 0.471                                |
| 8.2      | Sliding Window (Fold 3)     | EBM     | 0.462                                | 0.469                             | 0.469                               | 0.435                                 | 0.435                             | 0.435                                |
|           |                             | Ordered | 0.430                                | 0.449                             | 0.449                               | 0.422                                 | 0.416                             | 0.416                                |
| 8.3      | Expanding Window            | EBM     | 0.491                                | 0.445                             | 0.445                               | 0.469                                 | 0.452                             | 0.452                                |
|           |                             | Ordered | 0.485                                | 0.463                             | 0.463                               | 0.418                                 | 0.321                             | 0.321                                |
| 8.3      | Expanding Window (Fold 2)   | EBM     | 0.464                                | 0.472                             | 0.472                               | 0.469                                 | 0.457                             | 0.457                                |
|           |                             | Ordered | 0.478                                | 0.500                             | 0.500                               | 0.474                                 | 0.481                             | 0.481                                |
| 8.3      | Expanding Window (Fold 3)   | EBM     | 0.458                                | 0.457                             | 0.457                               | 0.454                                 | 0.452                             | 0.452                                |
|           |                             | Ordered | 0.434                                | 0.458                             | 0.458                               | 0.411                                 | 0.411                             | 0.411                                |
| 8.4      | Simple Split                | EBM     | 0.427                                | 0.441                             | 0.441                               | 0.471                                 | 0.419                             | 0.419                                |
|           |                             | Ordered | 0.413                                | 0.417                             | 0.417                               | 0.470                                 | 0.430                             | 0.430                                |
| 8.5      | Sliding Window              | EBM     | 0.481                                | 0.481                             | 0.481                               | 0.480                                 | 0.478                             | 0.478                                |
|           |                             | Ordered | 0.414                                | 0.357                             | 0.357                               | 0.408                                 | 0.291                             | 0.291                                |
| 8.5      | Sliding Window (Fold 2)     | EBM     | 0.476                                | 0.478                             | 0.478                               | 0.470                                 | 0.463                             | 0.463                                |
|           |                             | Ordered | 0.477                                | 0.491                             | 0.491                               | 0.466                                 | 0.469                             | 0.469                                |
| 8.5      | Sliding Window (Fold 3)     | EBM     | 0.455                                | 0.472                             | 0.472                               | 0.442                                 | 0.444                             | 0.444                                |
|           |                             | Ordered | 0.428                                | 0.447                             | 0.447                               | 0.420                                 | 0.412                             | 0.412                                |
| 8.6      | Expanding Window            | EBM     | 0.482                                | 0.481                             | 0.481                               | 0.479                                 | 0.477                             | 0.477                                |
|           |                             | Ordered | 0.414                                | 0.357                             | 0.357                               | 0.408                                 | 0.291                             | 0.291                                |
| 8.6      | Expanding Window (Fold 2)   | EBM     | 0.469                                | 0.478                             | 0.478                               | 0.465                                 | 0.455                             | 0.455                                |
|           |                             | Ordered | 0.482                                | 0.501                             | 0.501                               | 0.474                                 | 0.476                             | 0.476                                |
| 8.6      | Expanding Window (Fold 3)   | EBM     | 0.438                                | 0.448                             | 0.448                               | 0.439                                 | 0.445                             | 0.445                                |
|           |                             | Ordered | 0.433                                | 0.457                             | 0.457                               | 0.411                                 | 0.410                             | 0.410                                |
| 8.7      | Simple Split                | EBM     | 0.430                                | 0.433                             | 0.433                               | 0.480                                 | 0.459                             | 0.459                                |
|           |                             | Ordered | 0.404                                | 0.384                             | 0.384                               | 0.375                                 | 0.375                             | 0.373                                |
| 8.8      | Sliding Window              | EBM     | 0.501                                | 0.448                             | 0.448                               | 0.472                                 | 0.455                             | 0.455                                |
|           |                             | Ordered | 0.485                                | 0.423                             | 0.423                               | 0.420                                 | 0.375                             | 0.375                                |
| 8.8      | Sliding Window (Fold 2)     | EBM     | 0.468                                | 0.487                             | 0.487                               | 0.468                                 | 0.435                             | 0.435                                |
|           |                             | Ordered | 0.462                                | 0.479                             | 0.479                               | 0.454                                 | 0.463                             | 0.463                                |
| 8.8      | Sliding Window (Fold 3)     | EBM     | 0.458                                | 0.457                             | 0.457                               | 0.448                                 | 0.439                             | 0.439                                |
|           |                             | Ordered | 0.438                                | 0.451                             | 0.451                               | 0.440                                 | 0.424                             | 0.424                                |
| 8.9      | Expanding Window            | EBM     | 0.501                                | 0.448                             | 0.448                               | 0.472                                 | 0.455                             | 0.455                                |
|           |                             | Ordered | 0.491                                | 0.407                             | 0.407                               | 0.424                                 | 0.372                             | 0.372                                |
| 8.9      | Expanding Window (Fold 2)   | EBM     | 0.472                                | 0.483                             | 0.483                               | 0.461                                 | 0.440                             | 0.440                                |
|           |                             | Ordered | 0.472                                | 0.502                             | 0.502                               | 0.460                                 | 0.471                             | 0.471                                |
| 8.9      | Expanding Window (Fold 3)   | EBM     | 0.456                                | 0.456                             | 0.456                               | 0.445                                 | 0.433                             | 0.433                                |
|           |                             | Ordered | 0.444                                | 0.467                             | 0.467                               | 0.413                                 | 0.400                             | 0.400                                |

### Experiment 2: Hyperparameter Tuning

The aim of this series of experiments is to optimize model performance through performing various hyperparameter tuning using the best dataset fold from Experiment 1.

#### Sprint 9 Results

The table below illustrates the best hyperparameters discovered via manual tuning in Experiment 2.

| Fold | Model        | Best hyperparameters identified from experiments | Precision | Recall | Accuracy |
|------|--------------|---------------------------------------------|-----------|--------|----------|
| 1    | EBM          | outer_bags: 4, inner_bags: 5, learning_rate: 0.001 | 0.507     | 0.512  | 0.512    |
| 2    | OrderedModel | distr: "probit", method: "bfgs", max_iter: 50 | 0.522     | 0.545  | 0.545    |


#### Additional Experiment Results

Additionally, the table below lists more experiments conducted under Experiment 2.

**EBM**
Model experiments conducted using [dataset version 7.4 (fold 1)](#sprint-9-dataset-version-7).

| Model Hyperparameters                               | Precision | Recall | Accuracy |
|-----------------------------------------------------|-----------|--------|----------|
| outer_bags: 4, inner_bags: 5, learning_rate: 0.001  | 0.507     | 0.512  | 0.512    |
| outer_bags: 16, inner_bags: 5, learning_rate: 0.001 | 0.516     | 0.481  | 0.481    |
| outer_bags: 4, inner_bags: 10, learning_rate: 0.001 | 0.504     | 0.495  | 0.495    |
| outer_bags: 16, inner_bags: 10, learning_rate: 0.001| 0.518     | 0.484  | 0.484    |
| outer_bags: 4, inner_bags: 5, learning_rate: 0.005  | 0.505     | 0.497  | 0.497    |
| outer_bags: 16, inner_bags: 5, learning_rate: 0.005 | 0.526     | 0.483  | 0.483    |
| outer_bags: 4, inner_bags: 10, learning_rate: 0.005 | 0.513     | 0.492  | 0.492    |
| outer_bags: 16, inner_bags: 10, learning_rate: 0.005| 0.520     | 0.479  | 0.479    |
| outer_bags: 4, inner_bags: 5, learning_rate: 0.03   | 0.510     | 0.505  | 0.505    |
| outer_bags: 16, inner_bags: 5, learning_rate: 0.03  | 0.528     | 0.483  | 0.483    |
| outer_bags: 4, inner_bags: 10, learning_rate: 0.03  | 0.514     | 0.487  | 0.487    |
| outer_bags: 16, inner_bags: 10, learning_rate: 0.03 | 0.522     | 0.479  | 0.479    |

**OrderedModel**
Model experiments conducted using [dataset version 7.4 (fold 2)](#sprint-9-dataset-version-7).

| Model Hyperparameters                               | Precision | Recall | Accuracy |
|-----------------------------------------------------|-----------|--------|----------|
| distr: "probit", method: "bfgs", max_iter: 50       | 0.500     | 0.529  | 0.529    |
| distr: "probit", method: "bfgs", max_iter: 70       | 0.497     | 0.529  | 0.529    |
| distr: "probit", method: "bfgs", max_iter: 100      | 0.494     | 0.527  | 0.527    |
| distr: "logit", method: "bfgs", max_iter: 50        | 0.522     | 0.539  | 0.539    |
| distr: "logit", method: "bfgs", max_iter: 70        | 0.493     | 0.525  | 0.525    |
| distr: "logit", method: "bfgs", max_iter: 100       | 0.488     | 0.521  | 0.521    |
