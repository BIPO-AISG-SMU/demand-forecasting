# Unit Tests

## Overview
The main objective of unit testing is to isolate written code to test and determine if it works as intended. When implemented correctly, unit tests can detect early flaws in code which may be more difficult to find in later testing stages.

Unit testing is a component of test-driven development (TDD) methodology that takes a meticulous approach in building a product by means of continual testing and revision. This testing method is also the first level of software testing, which is performed before other testing methods such as integration testing. Unit tests are typically isolated to ensure a unit does not rely on any external code or functions. Teams should perform unit tests frequently, either manually or more often automated.

## Unit Testing Coverage

Within the Demand Forecasting module, the unit tests covers the following modules in the pipeline:
- `data_loader`
- `data_preprocessing`
- `feature_engineering`
    - `encoding.py`
        - `_ordinal_encoding_transform_`
        - `_one_hot_encoding_transform_`
    - `lag_feature_generation.py`
        - `_create_simple_lags_`
        - `_create_sma_lags_`
    - `standardize_normalize.py`
        - `_standard_norm_transform_`
- `time_agnostic_feature_engineering`
    - `feature_indicator_creation.py`

## Running Pytest

Open up a terminal/command prompt, change directory to the project folder with the necessary conda/python environment containing pytest library activated. Example: 
```
(bipo-unit-test) <path/to/100E_projects_BIPO> pytest
```

A successful outcome should look like below when all tests have passed. For each test files script, there are only '.' represented indicating pass, where each . represents a test function tested.
```
======================================== test session starts =========================================================
platform linux -- Python 3.10.13, pytest-7.4.3, pluggy-1.0.0
rootdir: /polyaxon-v1-data/workspaces/zhiqiangquek/95_unit_test_feature_engr
configfile: pyproject.toml
plugins: cov-4.1.0, dash-2.13.0, anyio-3.7.1
collected 34 items                                                                                                                                                                                                
src/tests/pipelines/data_loader/test_nodes.py .......                                                           [ 20%]
src/tests/pipelines/data_preprocessing/test_nodes.py ...............                                            [ 64%]
src/tests/pipelines/feature_engineering/test_nodes.py ......                                                    [ 82%]
src/tests/pipelines/time_agnostic_feature_engineering/test_nodes.py ......                                      [100%]

.
.
.
===============================================34 passed in 19.79s====================================================
```
In case of test failure, we would see a 'F' indicated instead of '.' appearing for the relevant test files' test functions.
Details on the type of error and the results are also indicated.

```
======================================== test session starts =========================================================
platform linux -- Python 3.10.13, pytest-7.4.3, pluggy-1.0.0
rootdir: /polyaxon-v1-data/workspaces/zhiqiangquek/95_unit_test_feature_engr
configfile: pyproject.toml
plugins: cov-4.1.0, dash-2.13.0, anyio-3.7.1
collected 34 items                                                                                                                                                                                                
src/tests/pipelines/data_loader/test_nodes.py FFFFFFF                                                     
========================================================= FAILURES ===================================================
______________________________ TestErrorFail.test_error ______________________________________________________________

self = <test_nodes.TestErrorFail testMethod=test_error>

    def test_error(self):
>       raise Exception('oops')
E       Exception: oops

test_nodes.py:5 Exception
.
.
.
```

### Configuring Pytest

To configure Pytest to generate a coverage report after execution, you can install the [pytest-cov](https://pypi.org/project/pytest-cov/) plugin and configure it by adding the following lines to `pyproject.toml` in the repository root folder.

```
[tool.pytest.ini_options]
addopts = """
--cov-report term-missing \
--cov src/<package_name> -ra"""
```

A simple coverage report generated looks similar to the below:

```
---------- coverage: platform linux, python 3.10.13-final-0 ----------
Name                                                                                      Stmts   Miss  Cover   Missing
-----------------------------------------------------------------------------------------------------------------------
src/bipo/__init__.py                                                                          2      0   100%
src/bipo/__main__.py                                                                         31     31     0%   4-48
src/bipo/hooks/DataCleanupHooks.py                                                           32     15    53%   42-60
src/bipo/hooks/MemoryProfilingHooks.py                                                       20      7    65%   19, 36-44, 53-63
.
.
.
```

For more details, refer to Kedro's documentation on [Automated Testing](https://docs.kedro.org/en/stable/development/automated_testing.html). 

## Conventions in Writing Unit Tests 

1. All unit tests should be written in `test_nodes.py`. By default, when creating new Kedro pipeline via the command, `kedro pipeline create <pipeline name>`, only `test_pipeline.py` is created. You need to manually create `test_nodes.py` yourself.

2. All unit testing classes and functions must be named with a `Test` prefix and `test_` prefix respectively. Note that any classes or functions not named with the `Test` or `test_` prefixes would be ignored by pytest during execution, with no errors or warnings generated.

4. For each fixture used by a test function, there is typically a parameter (named after the defined fixture) in the test function’s definition. An example is shown as follows:

```
# Define fixture
@pytest.fixture(scope="module")
def create_preprocessing_datapoints(self):
    # Fixture function
    def _create_preprocessing_datapoints(
        datapoint_rows: List[Dict], index=None, columns=None
    ):
        return pd.DataFrame(datapoint_rows, index=index, columns=columns)

    return _create_preprocessing_datapoints

# Test function with fixture as argument
def test_create_min_max_feature_diff(self, create_preprocessing_datapoints):
.
.
.
```

### About Fixtures

Fixtures are used to feed some data to the tests such as database connections, URLs to test and some sort of input data.

Example of using a fixture for generating dataframe rows via a factory method:

```
@pytest.fixture(scope="module")
def create_preprocessing_datapoints(self):
    # Fixture function
    def _create_preprocessing_datapoints(
        datapoint_rows: List[Dict], index=None, columns=None
    ):
        return pd.DataFrame(datapoint_rows, index=index, columns=columns)

    return _create_preprocessing_datapoints
```

## Additional Resources
- [Pytest fixture](https://docs.pytest.org/en/6.2.x/fixture.html)
- [Effective Python Testing With Pytest](https://realpython.com/pytest-python-testing/)
- [A Beginner's Guide to Unit Tests in Python](https://www.dataquest.io/blog/unit-tests-python/)

> Note: Tests don’t have to be limited to a single fixture, either. They can depend on as many fixtures as you want, and fixtures can use other fixtures as well.