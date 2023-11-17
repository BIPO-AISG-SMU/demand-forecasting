# FAQ

## OrderedModel Training

### What are some challenges in training OrderedModel?

Challenges with OrderedModel training primarily revolve around overparameterization. Overparameterization can lead to models that are overly complex relative to the data, resulting in poor predictive performance on new, unseen data. Additionally, having too many parameters can also increase computational demands unnecessarily.

#### Example
In sprint 9, when the number of _tsfresh_ features (`tsfresh_num_features` in `parameters.yml`) was set to 20, the total feature count inflated to over 60. This significant increase in features caused OrderedModel to fail to run. The following error message was encountered:

```
24/10/2023 02:52 | kedro.pipeline.node | ERROR | Node 'model_train_train_model: train_model([ordered_model.model_specific_preprocessing_train,parameters,params:ordered_model]) -> [ordered_model.model_training_artefact]' failed with error: There should not be a constant in the model
```

As a temporary solution, we set the constants argument within the OrderedModel's fitting function to `False`:
~~~
model = OrderedModel(y_train,X_train,distr='probit', hasconst=False)
~~~
This action reduced the parameter space by not considering certain variables as constants, thereby alleviating the burden on the model's optimization algorithm. Refer [here](https://www.statsmodels.org/stable/examples/notebooks/generated/ordinal_regression.html#Using-formulas---no-constant-in-model) for more infomation. It is important to note that this was the only solution attempted due to time constraints. 

### How can one avoid overparameterization in OrderedModel training?

To mitigate overparameterization, consider the following strategies:
- **Feature Selection**: Select only the most relevant features for training your model.
- **Cross-Validation**: Employ cross-validation techniques to evaluate the model's ability to generalize to unseen data.

### Can the process of feature selection be automated?

Yes, the feature selection process can be automated. Python libraries such as [tsfresh](https://tsfresh.readthedocs.io/en/latest/) and [scikit-learn](https://scikit-learn.org/) provide  methods for automatically selecting the best features based on performance metrics.

### What are _tsfresh_ features and their role in OrderedModel training?

In the context of time series, _tsfresh_ is often used to generate a vast number of features calculated from the given time series data. For detailed information on the features supported by _tsfresh_, refer to the [tsfresh documentation](https://tsfresh.readthedocs.io/en/latest/text/list_of_features.html). These features are used to capture the underlying patterns of the time series for use in various ML models, including OrderedModel.

### What are some tips on tuning the number of _tsfresh_ features?

When adjusting the number of _tsfresh_ features:
- Start with the default settings and refine based on the model's initial performance.
- Incrementally include more features while monitoring performance.
- Set a performance target to guide when to stop adding features.
- Employ domain knowledge to keep only features pertinent to the specific problem area.

For an integrated approach to _tsfresh_ feature adjustment for this project, review the section on Time-Dependent Feature Engineering within the [Data Pipeline Documentation](./data-pipeline#6-time-dependent-feature-engineering).

## Training a Model on Entire Dataset

### How do I train a model on entire dataset without any data splits?

Based on the current implementation, splitting the data into training, validation, and test sets is essential for the data pipeline and training pipeline to execute successfully. Therefore, model training without data splitting is not possible out of the box.

However, we can suggest a quick manual solution to train the model without data split, by following these steps:
1. In `data_split.yml` under `simple_split`, set `validation_days` and `testing_days` to 0. In `parameters.yml`, ensure that `split_approach_source` is `simple_split`.
2. Under `../src/bipo/pipelines/feature_engineering/pipeline.py`, remove all nodes that apply transformation to testing and validation sets. For example:  
    ```
    # Comment out all transformation nodes for val and test set.
        feature_engineering_pipeline_instance = pipeline(
            [
            .
            .
            .            
                # node(  # Handles validation set
                #     func=apply_binning_transform,
                #     inputs=[
                #         "time_agnostic_feature_engineering_validation",
                #         "parameters",
                #         "binning_encodings_dict",
                #     ],
                #     outputs="feature_engineering_validation",
                #     name="feature_engr_apply_binning_transform_validation",
                # ),
                # node(  # Handles testing set
                #     func=apply_binning_transform,
                #     inputs=[
                #         "time_agnostic_feature_engineering_testing",
                #         "parameters",
                #         "binning_encodings_dict",
                #     ],
                #     outputs="feature_engineering_testing",
                #     name="feature_engr_apply_binning_transform_testing",
                # ),
            ])
    ```
3. Run the data pipeline.
    ```
    # Bash
    kedro run --pipeline=data_loader; kedro run --pipeline=data_pipeline

    # Powershell 
    kedro run --pipeline=data_loader && kedro run --pipeline=data_pipeline
    ```
    If you are running the data pipeline on a Docker container, you will need to rebuild the Docker image due to the code changes in step 2 above. 
    
    Note that this step will only generate output in `../data/05_feature_engineering/features_merged/training`. However, `model_spec_preprocess_pipeline` also require inputs from `../data/05_feature_engineering/features_merged/validation` and `../data/05_feature_engineering/features_merged/testing`.

4. After running data pipeline, create 2 empty dataframes with the columns from `Date` and `binned_proxyrevenue`. Save the dataframes as CSV files in `/05_feature_engineering/features_merged/testing/testing_fold1_simple_split_param.csv` and `/05_feature_engineering/features_merged/validation/validation_fold1_simple_split_param.csv`.
5. In `pipeline_registry.py`, create a new pipeline "model_specific_fe_training_pipeline" by adding `model_spec_preprocess_pipeline` and `model_eval_pipeline`. The "model_specific_fe_training_pipeline" should be:
    ```
    "model_specific_fe_training_pipeline": model_spec_preprocess_pipeline + model_train_pipeline  
    ```
6. Run `model_specific_fe_training_pipeline` and the model weights that is trained on the entire dataset will be seen in `models` folder.
    ```
    kedro run --pipeline=model_specific_fe_training_pipeline
    ```

## Adding a New Model to the Pipeline

To facilitate the possibility of introducing new models, the tables below provide an overview on the files that need to be updated. Note that filepaths are relative to the project root directory.

### Configuration Files
|Filepath|Purpose|
|---|---|
| **conf/base/** | Contains all `.yml` configuration files. |
| `parameters.yml` | System-wide adjustable pipeline parameters. |
| `constants.yml` | Constants defining default values for key pipeline parameters. |
| `catalog.yml` | Registry of all data sources available for use by the project. See [The Kedro Data Catalog](https://docs.kedro.org/en/stable/data/data_catalog.html) for a complete guide.|
| parameters/`model_training.yml` | Adjustable parameters specific to the `model_training` Kedro pipeline. |


### Python Scripts
|Filepath|Purpose|
|---|---|
|src/bipo/pipelines/model_specific_preprocessing/nodes.py|Python script containing defined functions that conducts specific processing required by each model. Example conversion of datatype from numpy to some other data types.<br/>Note that there is no specific preprocessing required by the EBM and OrderedModel.|
|src/bipo/pipelines/model_specific_preprocessing/pipeline.py|Python script used for constructing a Kedro model specific preprocessing pipeline based on functions defined in nodes.py with wrapper input/outputs.|
|src/bipo/pipelines/model_training/nodes.py|Python script containing defined functions used for mlflow setup and tracking, model training and related functionalities.|
|src/bipo/pipelines/model_training/pipeline.py|Python script used for constructing a Kedro model training pipeline based on functions defined in nodes.py with wrapper input/outputs.|

### Python Dependencies
|Filepath|Purpose|
|---|---|
|src/requirements.txt|Text file containing library dependencies of the project repository.|

### 1. Updating configuration files 

#### Model training hyperparameter config file: `model_training.yml`

To add configurable hyperparameters, you should append to the existing files based on the example shown below. Ideally, the names of hyperparameters should be the same as those used by the actual library.
```
new_model:
    params: {
        model_name: "new_model",
        hyperparameter1: value1,
        .
        .
        .
        hyperparameterN: valueN
    }
```

#### Data Catalog file: `catalog.yml`

There are 3 key sections in this file where additional data catalogs need to be added:
1. model-specific preprocessing;
2. model training; and
3. model evaluation

**1. Model-specific preprocessing**

In the event that you want to view the processed data after execution of model specific preprocessing to validate the correctness of the preprocessing logic, you should add the following 3 data catalogs representing the training, validation and testing dataset respectively. 

> Note: The statement `<<: \*catalog_partitioned_default` indicates the use of yml alias mechanism referencing the defined *catalog_partitioned_default* anchor located on the top of the file. The use of anchor is to facilitate the reusability of common settings required for the defined datasets of IncrementalDataSet. In the case where other dataset type is required, please refer to [Kedro Data Catalog](https://docs.kedro.org/en/stable/data/data_catalog.html).

```
new_model.model_specific_preprocessing_training:
  <<: *catalog_partitioned_default
  path: data/06_model_specific_preprocessing/new_model/training
  layer: ebm_model_specific_preprocessing_training
  
new_model.model_specific_preprocessing_validation:
  <<: *catalog_partitioned_default
  path: data/06_model_specific_preprocessing/new_model/validation
  layer: ebm_model_specific_preprocessing_validation
  
new_model.model_specific_preprocessing_testing:
  <<: *catalog_partitioned_default
  path: data/06_model_specific_preprocessing/new_model/testing
  layer: ebm_model_specific_preprocessing_testing
```

**2. Model training**

To configure the path which model artefacts is to be stored upon completion of training(assuming `.pkl` is used). Add in the following in model training section containing `ordered_model.model_training_artefact` and `ebm.model_training_artefact` parameters. This indicates that generated artefact should be stored in `/models` subdirectory of the project directory.
```
new_model.model_training_artefact:
  type: pickle.PickleDataSet
  filepath: models/new_model.pkl
  backend: pickle
  versioned: true
  load_args:
    encoding: 'utf-8'
```

**3. Model evaluation**

To capture the evaluation results of the new model on training,validation and testing dataset, add the following 3 data catalogs representing training/validation/testing datasets evaluated under the 'Evaluation' section in the file. To align and be consistent with the naming used for other models, you can use the example filepath and layer name as follows.

```
new_model.model_evaluation_training_result:
  type: json.JSONDataSet
  filepath: data/07_model_evaluation/new_model_train
  layer: new_model_evaluation_train

new_model.model_evaluation_validation_result:
  type: json.JSONDataSet
  filepath: data/07_model_evaluation/new_model_val
  layer: new_model_evaluation_val

new_model.model_evaluation_testing_result:
  type: json.JSONDataSet
  filepath: data/07_model_evaluation/new_model_test
  layer: new_model_evaluation_test
```

#### Parameters config file: `parameters.yml`

To use the new model for model training and evaluation, change the following `model` parameter in the file:
```
model: "new_model"
```

#### Constants config file: `constants.yml`

As the code checks for validness of specified model name to be used for model training, append the new model name to the `valid_model_name_list` parameter. If you wish, you can set `default_model` parameter to the new model in the next line. Subsequently, append the preferred hyperparameters defaults for `new_model`.
```
modeling:
    valid_model_name_list: ["ebm", "ordered_model", "new_model"]
    default_model: "new_model"
    .
    .
    .
    new_model:
        hyperparameter1: value1,
        .
        .
        .
        hyperparameterN: valueN
```

### 2. Updating Python scripts

The key scripts to be updated are `nodes.py` and `pipeline.py`, where the latter imports functions defined in the former to conduct a sequence of operations. 

Useful resources for quick understanding:
- [Kedro documentation](https://docs.kedro.org/en/stable/nodes_and_pipelines/index.html)
- [Data Science pipeline by Neptune.ai](https://neptune.ai/blog/data-science-pipelines-with-kedro)

#### Model specific preprocessing node: `src/bipo/pipelines/model_specific_preprocessing/nodes.py`

To facilitate special preprocessing steps, you would need to create new function(s) in `nodes.py` as per below. Inputs and outputs to the function would largely depend on where the function should be executed in the pipeline as defined in `pipeline.py`. 

#### Model specific preprocessing pipeline: `src/bipo/pipelines/mode_specific_preprocessing/pipeline.py`

The following processes are applicable in the model specific preprocessing pipeline, regardless of model:
- Removal of rows containing at least one null value and data features configured through "fe"-prefixed parameters in `conf/base/parameters.yml` for the purpose of encoding and binning feature engineering.
- Removal of single unique value features across dataset based on training dataset.
- Combining training/validation/testing datasets as single entity per fold to facilitate predictors-predicted features split, i.e. X-y split.

In the event where additional model specific preprocessing is required, it is likely that the node containing such functions should be inserted in between the two nodes as shown below.

```
node(
    func=remove_const_column,
    inputs=[
        "removed_columns_rows_testing",
        "constant_column_params",
    ],
    outputs="remove_const_colum_data_testing",
    name="model_spec_preprocess_remove_const_column_testing",
),

#ADDITIONAL PROCESSING NODES LIKELY TO BE INSERTED HERE

node(
    func=reorder_data,  # Combine train/val
    inputs=[
        "remove_const_colum_data_training",
        "remove_const_colum_data_validation",
        "remove_const_colum_data_testing",
    ],
    outputs="reordered_data_folds",
    name="model_spec_preprocess_reorder_data",
),
```

For any insertion of node functions in between nodes, please ensure that the input and output dependencies are specified correctly. This means input to the new node should be mapped to the output of other nodes executed previously and the output of new node serve as the input to the downstream nodes.

> Note: Any inputs or outputs specified that are not found in `conf/base/catalog.yml` would be treated as interim dataset generated on the fly (`MemoryDataSet`, in Kedro terms) and would be lost when pipeline runs are completed or terminated.

#### Model training node: `src/bipo/pipelines/model_training/nodes.py`

To include the new model in the training pipeline, you would want to introduce an additional `elif` statement in between the existing `if-else` block as follows.

```
if model_name == "ebm":
    try:
    .
    .
    .

# For new model
elif model_name == "new_model":
    try:
        hyperparameter1 = model_params_dict["hyperparameter1"]
        
# For ordered model
else:
    # Extract information from params dict
    try:
```

#### Model training pipeline: `src/bipo/pipelines/model_training/pipelines.py`

There is no need to make changes to the existing file if a new model **without any additional explanability processing** is to be introduced, and changes are made in accordance to the above suggestion through the use of introducing `elif` block statements. 

In the event where some form of explanability function is to be introduced for the new model(s), you may want to enhance the existing `explain_ebm()` function defined to cater for such changes and to utilise the model artefact and training data to facilitate the explanability process. In such situation, the `if-else` block in `pipeline.py` shown below needs to be modified to suit the new implementation.

```
if conf_params["model"] == "ebm" and conf_params["enable_explainability"]:
    pipeline_instance = pipeline_instance.only_nodes_with_tags(
        "model_training", "enable_explainability"
    )
else:
    pipeline_instance = pipeline_instance.only_nodes_with_tags("model_training")
```

### 3. Updating `requirements.txt`

For a quick update on `requirements.txt`, you may run the following command in your **activated conda environment**. Note that this will include all dependencies installed.

```
pip list --format=freeze > requirements.txt
```

In the event you do not wish to include dependencies of the core libraries, you may install and use the open source library [pip-chill](https://github.com/rbanffy/pip-chill) to help generate updated libraries without dependencies.

## Automated Hyperparameter Search

### What is hyperparameter search and why is it important?
Hyperparameter search is the practice of selecting the ideal combination of hyperparameters that govern the learning process of a machine learning algorithm. These settings, unlike model parameters, are not derived from the training data but are set prior to the learning process and directly influence model performance.

Optimal hyperparameter settings are crucial as they can significantly enhance model accuracy and efficiency, leading to superior outcomes for the given machine learning challenge.

### Can hyperparameter search be automated?
Indeed, the hyperparameter search can be automated. Tools such as Optuna, and Scikit-learn's GridSearchCV and RandomizedSearchCV, provide sophisticated methods for automating this process, streamlining model optimisation. 

### Which hyperparameter search technique should I use?
The choice of technique often depends on the complexity of the model and the computational resources available. 

| Technique | Ideal Usage Scenario | Description |
| :- | - | - |
| [GridSearchCV (Scikit-learn)](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)       | Optimal for models with fewer hyperparameters where exhaustive search is viable, prioritising precision over speed.                             | Systematically probes every combination within a predefined hyperparameter grid, ensuring the identification of the best parameters within the grid's boundaries. |
| [RandomizedSearchCV (Scikit-learn)](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html) | Suited to larger hyperparameter domains where exhaustive search is impractical. Good for a rapid overview with limited computational resources. | Selects hyperparameter sets at random from a defined distribution. This can lead to faster but potentially less precise results compared to GridSearchCV.         |
| [Optuna](https://optuna.readthedocs.io/en/stable/)                            | Recommended for complex and high-dimensional spaces, with the advantage of successive trials informing subsequent search paths.                 | Employs Bayesian optimization to traverse the hyperparameter space efficiently, concentrating on areas indicated as promising by past trial outcomes.             |

### How to evaluate the effectiveness of a hyperparameter search?
The effectiveness is typically evaluated by the improvement in the model's performance on a validation set. Consider to monitor the performance metrics relevant to your specific problem, such as: 
- Accuracy, precision, recall or F1 score for classification tasks; or
- Mean squared error (MSE), root mean square error (RMSE) or mean absolute error (MAE) for regression tasks.

Additionally, consider the time and computational resources used by the search process.

### How do I determine the range of values to explore in a hyperparameter search?
Determining the value range for hyperparameters can be guided by past empirical evidence, expertise in the relevant domain, or by initiating with a broad spectrum and iteratively refining this based on initial search findings.
