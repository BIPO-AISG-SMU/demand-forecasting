# Overview of Kedro (v0.18.11)

## What is Kedro?
Kedro is an open-source Python framework for creating reproducible, maintainable, and modular data science code. It borrows concepts from software engineering best practices and applies them to machine learning code. More information on Kedro is available at their [documentation page](https://kedro.readthedocs.io/en/stable/).

## Elements of Kedro
- [Project Template](https://docs.kedro.org/en/stable/kedro_project_setup/starters.html)
  - Inspired by [cookie-cutter data science](https://drivendata.github.io/cookiecutter-data-science/) template
  - Built-in support for Python logging, pytest for unit tests and Sphinx for documentation
- [Data Catalog](https://docs.kedro.org/en/stable/data/index.html)
- [Nodes and Pipelines](https://docs.kedro.org/en/stable/nodes_and_pipelines/index.html)
  - Constructs which enable a data-centric workflow
- [Kedro-Viz](https://docs.kedro.org/en/stable/visualisation/index.html)
- Experiment Tracking
  - Paired with [MLFlow](https://mlflow.org/)
  - Visualization within Kedro-Viz
- [Hooks](https://docs.kedro.org/en/stable/hooks/index.html)

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

### Pipeline Registry

Kedro pipelines must be defined under the [pipeline registry](../src/bipo/pipeline_registry.py) in order to use the pipeline.

An example is shown below:

```python
from bipo.pipelines import data_processor as dl
from kedro.pipeline import Pipeline


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

> **Note:** Pipelines can be chained together to be ran sequentially. Pipelines specified within the "_\_default__" pipeline will be executed upon execution of `kedro run`.

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

