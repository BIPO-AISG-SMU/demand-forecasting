# Inference Module

This document provides detailed information on the inference module. Ensure you've followed the [`Environment Setup`](2-setup.html) before diving in.

## Overview

The Inference Module is responsible for executing model predictions. Given input data via the API request, this module preprocesses the data, feeds it into the trained model and produces the corresponding predictions. These predictions can be utilised in various downstream applications, analytics, or reporting tools. 

## Pipeline Configuration

The module relies on configuration settings to define its behaviour and integrate with other components. The primary configuration for the application is defined in the `src/bipo_fastapi/config.py` file. 

When initializing, the module first checks for the presence of environment variables corresponding to each setting. If an environment variable is present, its value is used; otherwise, the default value from `config.py` is taken.

| Parameter                | Type      | Description                            | Environment Variable       | Default Value                            |
| ------------------------ | --------- | -------------------------------------- | -------------------------- | ---------------------------------------- |
| API_NAME                 | str       | The name of the API                    | API_NAME                   | BIPO FastAPI                             |
| API_VERSION              | str       | The version of the API                 | API_VERSION                | /api/v1                                  |
| INTERMEDIATE_OUTPUT_PATH | str       | Path to output intermediate files      | INTERMEDIATE_OUTPUT_PATH   | ../data/10_model_inference_output        |
| LOGGER_CONFIG_PATH       | str       | Path to the logging configuration file | LOGGER_CONFIG_PATH         | ../conf/base/logging.yml                 |
| PRED_MODEL_PATH          | str       | Path to the prediction model file      | PRED_MODEL_PATH            | ../models/orderedmodel_prob_20230816.pkl |
| PRED_MODEL_UUID          | str       | UUID of the prediction model           | PRED_MODEL_UUID            | 0.1                                      |
| SALES_CLASS_NAMES        | List[str] | Names of the sales classes             | (Derived from `config.py`) | ["Low", "Medium", "High", "Exceptional"] |
