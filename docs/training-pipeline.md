# Training Pipeline

## Overview
This guide outlines the architecture and components of the training data pipeline, designed to be modular, flexible, and easily adaptable to either Ordered Model or Explainable Boosting Machine. It is divided mainly into Model Training and Model Evaluation tasks.

## Pipeline Design

### Design Considerations
1. Modular Design: Each component is modular for easy maintenance and updates.
2. Decoupling: Pipeline stages are designed to work independently, allowing for easier testing and modifications.
3. Configurability: Most components are configurable through config files, allowing for easy adjustments.

### Flowchart
The diagram below provides a high-level overview of the training pipeline's process flow.

![Alt text](assets/training-pipeline.png)

1. Model input data from `data/06_model_specific_preprocessing` is used in model training and model evaluation module.
2. Pipeline parameters and model hyperparameters can be configured in `parameters.yml` and `model_training.yml` respectively.
3. Model Training module will generate model weights that is stored in `models` folder.
4. Model Evaluation module will use model weights from `models` folder and data from `data/06_model_specific_preprocessing` to generate evaluation matrics stored in `data/07_model_evaluation`
5. (optional) If ML Flow is enabled, MLflow artefacts (model parameters and metrics) will be stored in `mlruns` folder.

### Choice of Models

| Model | Description | Repository |
| --- | --- | --- |
| Ordered Model | Ordered Model uses ordinal regression which is a statistical technique that is used to predict behavior of ordinal level dependent variables with a set of independent variables. The dependent variable is the order response category variable and the independent variable may be categorical or continuous. | [GitHub](https://github.com/statsmodels/statsmodels/) |
| Explainable Boosting Machine (EBM) | Explainable Boosting Machine (EBM) is a tree-based, cyclic gradient boosting Generalized Additive Model with automatic interaction detection. EBMs are often as accurate as state-of-the-art blackbox models while remaining completely interpretable. Although EBMs are often slower to train than other modern algorithms, EBMs are extremely compact and fast at prediction time. | [GitHub](https://github.com/interpretml/interpret) |

## Training Pipeline Configuration

### Key Parameters in the `parameters.yml` File
| Parameter | Type | Description | Default Value |
| --- | --- | --- | --- |
| model | `str` | Choice of model to train. 2 options: `"ebm"` or `"ordered_model"` | `ebm` |
| split_approach_source | `str` | Choice of cross validation data splits used. Note: this split approach should be the same as specified during `data pipeline`. 3 options: `"simple_split"`,`"expanding_window"` or `"sliding_window"` | `simple_split` |
| fold | `int` | Cross validation fold to be used for model training and evaluation. Note that `"simple_split"` only has 1 fold, hence set fold as 1. | `fold` |
| enable_mlflow | `Bool` | `True` to enable ML Flow tracking, `False` to disable. | `True` |
| is_remote_mlflow | `Bool` | `True` to enable remote ML Flow tracking, `False` to disable. | `False` |
| tracking_uri | `str` | HTTP Server hosting an MLflow tracking server. | `http://10.43.130.112:5005` |
| experiment_name_prefix | `str` | Name of model experiment | `bipo` |

### Model Hyperparameters in the `model_training.yml` File

#### OrderedModel 

| Parameter | Type | Description | Default Value |
| --- | --- | --- | --- |
| distr | `str` | `probit` for normal distribution or `logit` for logistic distribution. Refer [HERE](https://www.statsmodels.org/dev/generated/statsmodels.miscmodels.ordinal_model.OrderedModel.html#statsmodels.miscmodels.ordinal_model.OrderedModel-parameters) for more infomation. | `probit` |
| method | `str` | The method determines which solver from scipy.optimize is used. Refer [HERE](https://www.statsmodels.org/dev/generated/statsmodels.miscmodels.ordinal_model.OrderedModel.fit.html#statsmodels.miscmodels.ordinal_model.OrderedModel.fit) for more infomation.| `bfgs` |
| max_iter | `int` | The maximum number of iterations to perform. | `10` |

#### Explainable Boosting Machine (EBM) 

Refer to the [InterpretML API documentation](https://interpret.ml/docs/ExplainableBoostingClassifier.html) for the complete list of hyperparameters.

| Parameter | Type | Description | Default Value |
| --- | --- | --- | --- |
| outer_bags | `int` | Number of outer bags. Outer bags are used to generate error bounds and help with smoothing the graphs. | `8` |
| inner_bags | `int` | Number of inner bags. 0 turns off inner bagging. | `0` |
| learning_rate | `int` | Learning rate for boosting. | `0.01` |
| interactions | `int` | Interaction terms to be included in the model.| `0` |
| max_leaves | `int` | Maximum number of leaves allowed in each tree. | `3` |
| max_bins | `int` | Max number of bins per feature for the main effects stage | `256` |
| min_samples_leaf | `int` | Minimum number of samples allowed in the leaves. | `2` |

## Model Input Data Definition
The training pipeline accepts data generated from the `data pipeline` in `06_model_specific_preprocessing`. 

| Column | Type | 
| --- | --- | 
| maximum_temperature_c | `float` | 
| factor | `float` | 
| name_counts | `int` | 
| is_name_start | `int` | 
| is_name_end | `int` | 
| is_weekday | `int` | 
| is_school_holiday | `int` | 
| is_public_holiday | `int` | 
| is_daily_rainfall_total_mm | `int` | 
| is_pandemic_restrictions | `int` | 
| lag_9_proxyrevenue | `float` | 
| lag_14_proxyrevenue | `float` | 
| lag_9_sma_7_days_proxyrevenue | `float` | 
| lag_mean_1_week_proxyrevenue | `float` | 
| lag_mean_2_week_proxyrevenue | `float` | 

This data definition is accurate for the trained model weights released on 19 October 2023. 

## How to Run

The training pipeline can be executed using the two methods outlined below.

### 1. Batch Script (Docker Container)

#### Prerequisites
- Ensure `data`, `conf`, `models`, `logs` and `mlruns` folders are present in the directory where docker image is running.
    - If you want to train `ordered_model` or `ebm`, please ensure `06_model_specific_preprocessing/ordered_model` or `06_model_specific_preprocessing/ebm_model` folders are present.
- The correct `image_name` and `docker_registry` (optional) should be defined in `run_model_training.bat`.

#### What `run_model_training.bat` does
1. Pulls training pipeline Docker image from Docker registry (optional).
2. Runs training pipeline Docker image and mount `data`, `conf`, `models`, `logs` and `mlruns` folders as persistent volumes.  
3. If ML Flow is installed, script will run `mlflow ui` on local machine and experiment tracking can be observed in `http://127.0.0.1:5000` on local machine. 

#### Executing the script

Run the following command in your terminal window to execute `run_model_training.bat`:

~~~powershell
# windows powershell
.\scripts\run_model_training.bat   

# linux bash
scripts/run_model_training.bat
~~~
You should see output similar to the below:
~~~
Status: Image is up to date for registry.aisingapore.net/100e-bipo/bipo-training-pipeline:0.1.2
registry.aisingapore.net/100e-bipo/bipo-training-pipeline:0.1.2
"Successfully pull image: registry.aisingapore.net/100e-bipo/bipo-training-pipeline:0.1.2"
"Starting docker run with image: registry.aisingapore.net/100e-bipo/bipo-training-pipeline:0.1.2 "
16cfe16e9f25993737c29b3b3ffb45c138a260d5c5e517dc27d37d711b4e7dfe
"Training Pipeline Docker Container run successfully"
"mlflow is installed on your local environment. Running mlflow"
INFO:waitress:Serving on http://127.0.0.1:5000
~~~

### 2. Kedro Command

#### Prerequisites
- Ensure that Kedro is installed in the Conda environment on your local machine.
- `data/06_model_specific_preprocessing` files are required.

#### Executing the script

Run the following commands in your terminal window, which specifies the `pipeline` argument as `training_pipeline` to run the training pipeline.

~~~python
kedro run --pipeline=training_pipeline
~~~
You should see output similar to the below:
~~~
18/10/2023 12:46 | kedro.framework.session.session | INFO | Kedro project kedro_docker
18/10/2023 12:46 | kedro.io.data_catalog | INFO | Loading data from 'ebm.model_specific_preprocessing_validation' (PartitionedDataset)...
.
.
.
18/10/2023 12:47 | kedro.io.data_catalog | INFO | Saving data to 'ebm.model_evaluation_val_result' (JSONDataSet)...
18/10/2023 12:47 | kedro.runner.sequential_runner | INFO | Completed 7 out of 7 tasks
18/10/2023 12:47 | kedro.runner.sequential_runner | INFO | Pipeline execution completed successfully.
~~~

## Model Evaluation

The table below summarises the metrics logged by the training pipeline during the model evaluation stage.

| Metrics | Description |
| --- | --- |
| Accuracy | The measure of the overall correctness of a model. It represents the ratio of correctly predicted instances to the total instances in the dataset. |
| Precision | The measure of the accuracy of positive predictions made by the model. It's the ratio of true positive predictions to all positive predictions made by the model |
| Recall | Recall measures the ability of a model to identify all relevant instances, specifically true positives. It's the ratio of true positives to all actual positive instances in the dataset. |
| F1 Score | Harmonic mean of precision and recall. It provides a balanced measure of model performance, taking into account both false positives and false negatives. |

## File Structure in Container

The following file structure depicts the key directories and files contained in the containerised training submodule. **This is not the same directory with the host computer on which Docker is running.**

```
├──/app/bipo_demand_forecasting/
    ├── pyproject.toml
    ├── conf/ (mounted)
    ├── data/ (mounted)
    ├── mlruns/ (mounted)
    ├── models/ (mounted)
    ├── logs/ (mounted)
    │   ├── info.log
    │   └── error.log
    └── src/
        ├── bipo/
        │   ├── pipelines/
        │   │   └── ...
        ├── requirements.txt
        └── utils.py
```
