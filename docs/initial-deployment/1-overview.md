# BIPO Demand Forecasting Module: Project Overview

## Purpose of Documentation

This documentation serves as a guide for the technical user in deploying and using the inference submodule on a typical Linux system. 

The reader is expected to have familiarity with using the command line and the Linux environment.

This guide does not provide any information pertaining to:
- Training submodule
- Data pipeline submodule
- Deployment on cloud

## 1. Architecture

## 1.1. Solution Architecture
This diagram shows the complete architecture of the Demand Forecasting module.
![image](./assets/ml-pipeline.png)

## 1.2. Initial Deployment Architecture
The following diagram illustrates how an external system may interact with the Demand Forecasting inference submodule via the REST API, as well as the overall internal structure of the submodule.
![image](./assets/Initial_deployment_architecture.png)

## 2. Deployment Package Contents
```
├── bipo_demand_forecasting/
    ├── conf/ (Created and to be bind mounted)
    |    └── base/
    |        ├──parameters.yml
    |        ├──logging.yml
    |        └──constants.yml
    ├── data/ (Created and to be bind mounted)
    ├── logs/ (Created and to be bind mounted)
    ├── models/ (Created and to be bind mounted)
    └── docker/(Created)
```

### 2.1. Configuration

| Path | Description |
| :- | - |
| **conf/base/** | All .yml configurations will be in this folder. |
| parameters.yml | Adjustable parameters |
| logging.yml | Logging configuration |
| constants.yml | Defined constants for use in python script. (Should not touch) |

### 2.2 Data

| Path | Description |
| :- | - |
| **data/** | All test data there will be used for inference and will be in this folder|
|01_raw/|Subdirectories for ...|
|02_data_loading/|Subdirectories for ...|
|03_data_preprocessing/|Subdirectories for ...|
|10_mode/|Subdirectories for ...|

### 2.3 Logs

| Path | Description |
| :- | - |
| **logs/** | All logs generated in the project will be in this folder.|
|info.log|Logs events that occurred in general|
|error.log|Logs events with errors encountered|

### 2.4. Docker image

The docker image will be provided as a `.tar.gz` archive, named `bipo_inference_initial.tar`.

| Path | Description |
| :- | - |
| **docker/** | All docker archive files will be in this folder.  Containerised image are exported into `.tar.gz` archive files and stored here for loading into AWS ECR.|

### 2.5. Trained model file(s)

The trained model file would be stored in an AWS S3 bucket. To ensure successful download, necessary S3 permissions are required to be configured in AWS Roles settings by the AWS administrator. 

| Path | Description |
| :- | - |
| **models/** | All trained models trained/saved as pickle format used in the project will be in this folder, with `.pkl` as extension. |
|`orderedmodel_20230818.pkl`|Trained model for inferencing. Model supported for initial deployment is *OrderedModel* from the statsmodels library|

### 2.6. OS port usage

Any firewall configurations or other applications installed should not be using the stated ports:

| Port | Description |
| :- | - |
| 2375 | Docker unencrypted communication |
| 2376 | Docker encrypted communication |
| 8000 | FastAPI endpoint|

## 3. Project File Structure in Container

The following file structure depicts the key directories and files contained in the containerised inference pipeline. **This is not the same directory with the host computer which docker is running.**

```
├──/app/bipo_demand_forecasting/
    ├── conf/ (mounted)
    ├── data/ (mounted)
    ├── models/ (mounted)
    ├── logs/ (mounted)
    │   ├── info.log
    │   └── error.log
    └── src/
        ├── bipo/
        │   ├── inference_pipeline/
        │   │   ├──data_preprocessing.py
        │   │   ├──feature_engineering.py
        │   │   ├──model_specific_fe.py    
        │   │   └──inference_pipeline.py
        │   ├── pipelines/
        │   │   └── ...
        ├── bipo_fastapi/
        │   ├── v1/ (model inference)
        |   │   └── routers/ (Code separation into multiple files)
        │   ├── config.py
        │   ├── deps.py
        │   ├── logs.py
        │   ├── main.py (entrypoint for infernence)
        │   ├── schemas.py
        │   └── ...
        ├── requirements.txt
        └── utils.py
```

