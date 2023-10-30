# Overview of Kedro (v0.18.11)

## What is Kedro?

Kedro is an open-source Python framework for creating reproducible, maintainable, and **modular** data science code. It borrows concepts from software engineering best practices and applies them to machine learning code. More information on Kedro is available in its [documentation](https://kedro.readthedocs.io/en/stable/).

## Elements of Kedro
![kedro overview](./assets/kedro_overview.png)

Although Kedro has numerous feature and functionalities, not all of them are used in this project. The key elements utilised in the context of the Demand Forecasting module are:

- [Project Template](https://docs.kedro.org/en/stable/kedro_project_setup/starters.html)
  - Inspired by [cookie-cutter data science](https://drivendata.github.io/cookiecutter-data-science/) template.
  - Built-in support for Python logging and Sphinx for documentation.
- [Configuration](https://docs.kedro.org/en/0.18.5/kedro_project_setup/configuration.html)
  - Base folder for default settings. 
  - Local folder for user specific config like IDE settings, security credentials. 
  - Project config is located in [base](../conf/base/) folder, consisting of YAML files. 
- [Data Catalog](https://docs.kedro.org/en/stable/data/index.html)
  - Registry of all data sources, and specified with a [catalog.yml](../conf/base/catalog.yml) file.
  - Used as node inputs/outputs  
  - Specify the dataset type, filepath, load and save arguments 
- [Datasets](https://docs.kedro.org/en/stable/kedro_datasets.html#module-kedro_datasets)
  - CSVDataSet - uses Pandas backend
  - JSONDataSet - for artefact
  - PickleDataSet - for artefact
  - MemoryDataset - in-memory Python object (not defined in data catalog)
  - IncrementalDataset - inherits from PartitionedDataset which loads/saves each partition that is stored as a separate file within a directory.
  ```
  # IncrementalDataset will load/save all the separate files in a directory. Here, it is all the processed outlet csv files in 03_data_preprocessing
  ├── data
  │   └── 03_data_preprocessing
  |       └── proxy_revenue_201_processed.csv
  |       └── proxy_revenue_202_processed.csv
  |       ...
  ```
- [Nodes](https://docs.kedro.org/en/stable/nodes_and_pipelines/nodes.html)
  - Functions which are the building blocks of pipelines.  
- [Pipelines](https://docs.kedro.org/en/stable/tutorial/create_a_pipeline.html)
  - Constructs which are made up of nodes, and enables a data-centric workflow.
  - [Namespace](https://docs.kedro.org/en/0.18.0/tutorial/namespace_pipelines.html)
    - Instantiate the same pipeline structure multiple times, but provide different inputs/outputs.
- [Hooks](https://docs.kedro.org/en/stable/hooks/index.html)
  - Injects additional behavior at certain points of the pipeline. (E.g before/after a pipeline is run; before/after a node is run) 
  - [DataCleanupHooks](../src/bipo/hooks/DataCleanupHooks.py) cleans up the respective data folder before a pipeline is run, which ensures each pipeline is run on a clean slate without any old data remaining from the previous run.
  - Hooks are registered in [settings.py](../src/bipo/settings.py) 

Additional features:

- Visualization within [Kedro-Viz](https://docs.kedro.org/en/stable/visualisation/kedro-viz_visualisation.html) which shows a flowchart of pipelines

## Configuration
- Can be loaded directly as a node input.
  ```
  # src/bipo/pipelines/model_training/pipeline.py
  from kedro.pipeline.modular_pipeline import pipeline
  from kedro.pipeline import node

  pipeline_instance = pipeline(
        [
            node(
                func=train_model,  # In nodes.py
                inputs=[
                    "model_specific_preprocessing_train",
                    "parameters",  # parameters.yml file is used as node input
                    "params:model_namespace_params_dict", # reads files inside conf/base/parameters folder
                ],
                outputs="model_training_artefact",
                name="model_train_train_model",
                tags="model_training",
            ),
        ],
    )

  # src/bipo/pipelines/model_training/nodes.py
  def train_model(
    partitioned_input: Dict[str, pd.DataFrame],
    params_dict: Dict[str, Any], # parameters.yml 
    model_namespace_params_dict: Dict[str, Any], # reads files inside conf/base/parameters folder
  ):

  # instantiate parameters from config
    target_feature_name = params_dict["fe_target_feature_name"]
    model_params_dict = model_namespace_params_dict["params"]
    model_name = str(model_params_dict["model_name"])
  ```

- Can be loaded in a script.
  ```
  from bipo import settings # bipo/settings.py
  from kedro.config import ConfigLoader

  conf_loader = ConfigLoader(conf_source=settings.CONF_SOURCE)

  # loads config files with names that start with parameters or are located inside a folder with name starting with parameters.
  conf_params = conf_loader["parameters"] 

  # loads config files with names that start with constants
  conf_const = conf_loader.get("constants*") 

  # instantiate parameters
  model = conf_params["model"] 
  ```
## Nodes 
- arguments consist of func, inputs, outputs, name, tags.
- use of node tags allows selection of specific nodes to be run. 
```
# src/bipo/pipelines/model_training/pipeline.py
from kedro.pipeline.modular_pipeline import pipeline
from kedro.pipeline import node

# to be used as pipeline namespace
model = conf_loader["parameters"]["model"]  # Name of model (ordered_model or ebm)

pipeline_instance = pipeline(
        [
            node(
                func=train_model,  # function in nodes.py
                # inputs which are either string or list of string
                inputs=[    
                    "model_specific_preprocessing_train", # defined in data catalog
                    "parameters",  # parameters.yml file is used as node input
                    "params:model_namespace_params_dict",
                ], 
                # outputs which are either string or list of string.
                outputs="model_training_artefact",  # defined in data catalog
                name="model_train_train_model", # name of node
                tags="model_training",  # can enable only nodes with certain tags to be run. 
            ),
            node(
                func=explain_ebm,  # In nodes.py
                inputs=[
                    "model_training_artefact",
                    "model_specific_preprocessing_train",
                ],
                outputs=None,
                name="explain_ebm",
                tags="enable_explainability",
            ),
        ],
    )


    # runs both nodes in the pipeline_instance by selecting both "model_training", "enable_explainability" tags  
    if conf_params["model"] == "ebm" and conf_params["enable_explainability"]:
        pipeline_instance = pipeline_instance.only_nodes_with_tags(
            "model_training", "enable_explainability"
        )
    # only run the first node with tag "model_training"
    else:
        pipeline_instance = pipeline_instance.only_nodes_with_tags("model_training")
```
## [Pipeline Namespace](https://docs.kedro.org/en/0.18.0/tutorial/namespace_pipelines.html)
- arguments consist of pipe, inputs, outputs, parameters, namespace.
- namespace is added as a prefix to node inputs and outputs within the pipeline. 
- inputs and outputs are used to specify the respective inputs and outputs which we want to exclude adding the namespace prefix.   

Uses of namespace:
1. same pipeline can be used with different sets of parameters.
2. load/save different datasets specified in the data catalog.  

```
# src/bipo/pipelines/data_split/pipeline.py

# modular pipeline for the 3 data split approaches (simple_split, expanding_window, sliding_window)

def create_pipeline(**kwargs) -> Pipeline:
    # New pipeline for data split to add all subpipelines
    pipeline_instance = pipeline(
        [
            node(
                func=prepare_for_split,  # In nodes.py
                inputs=["data_merge", "params:data_split_params"],
                outputs="split_params_dict",
                name="data_split_prepare_for_split",
            ),
            node(
                func=do_time_based_data_split,  # In nodes.py
                inputs="split_params_dict",
                outputs="data_split",
                name="data_split_do_time_based_data_split",
            ),
        ],
    )

    # Instantiate multiple instances of pipelines with static structure, but dynamic inputs/outputs/parameters. Inputs/outputs required if not managed by namespace

    # simple split pipeline
    simple_split_pipeline = pipeline(
        pipe=pipeline_instance,
        inputs=["data_merge"],   
        parameters={
            "params:data_split_params": "params:simple_split",
        },    
        namespace="simple_split",
    )

    # parameters argument: overwrite data_split_params with simple_split. This looks for config files under the conf/base/parameters folder and searches for the key. In this case, the key is {simple_split} which is found in parameters/data_split.yml

    # namespace argument: will be added as a prefix to node inputs and outputs. Since data_merge is specified in the inputs argument, it will be excluded from the addition of namespace prefix. So, the resulting inputs and outputs modified by namespace are simple_split.split_params_dict and simple_split.data_split. 
  
  # expanding_window_pipeline and sliding_window pipeline are similar, just with a change in namespace which is their respective split approach. 
```


## Creating Kedro Pipelines

To create a pipeline, e.g. a `data_processor` pipeline, run the following command in the terminal:

```shell
kedro pipeline create data_processor
```

The above command generates all the files you need to start to write a `data_processor` pipeline:
- `nodes.py` and `pipeline.py` in the `src/bipo/pipelines/data_processor` folder for the main node functions that form your pipeline.
- `conf/base/parameters/data_processor.yml` to define the parameters used when running the pipeline.
- `src/tests/pipelines/data_processor` for tests for your pipeline.
- `__init__.py` files in the required places to ensure that the pipeline can be imported by Python.

The folder structure will look similar to the following:

```
├── README.md
├── conf
│   └── base
│       └── parameters
│           └── data_processor.yml
└── src
    ├── bipo
    │   ├── __init__.py
    │   └── pipelines
    │       ├── __init__.py
    │       └── data_processor
    │           ├── README.md
    │           ├── __init__.py
    │           ├── nodes.py
    │           └── pipeline.py
    └── tests
        ├── __init__.py
        └── pipelines
            ├── __init__.py
            └── data_processor
                ├── __init__.py
                └── test_pipeline.py
```

## Pipeline Registry

Kedro pipelines must be defined under the [pipeline registry](../src/bipo/pipeline_registry.py) in order to use the pipeline.

An example is shown below:

```python
from bipo.pipelines import data_processor as dl
from kedro.pipeline import Pipeline
from typing import Dict

def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    data_processor_pipeline = dl.create_pipeline()

    return {
      "__default__": pipeline([]),
      "data_processor": data_processor_pipeline
    }
```

> **Note:** Pipelines can be [chained](../src/bipo/pipeline_registry.py) together to be ran sequentially. Pipelines specified within the "_\_default__" pipeline will be executed upon execution of `kedro run`.

### Registry List

To get the list of all registered pipelines within the project, run the following command:

```shell
kedro registry list
```

This project currently consist of the pipelines listed below:

- `__default__`
- `training_pipeline`
- `data_loader`
- `data_pipeline`

## Running Specific Pipelines

```shell
kedro run --pipeline=<pipeline-name>
```

List of available pipelines could be found from the CLI command `kedro registry list`.

> **Note:** If you specify `kedro run` without the `--pipeline` option, it runs the `__default__` pipeline from the dictionary returned by `register_pipelines()`.

