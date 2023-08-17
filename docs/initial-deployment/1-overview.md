# Project Overview

## Purpose of Documentation

The purpose of the documentation is to provide an information overview pertaining to solution architecture and project file structure for inference module only. 

The documentation does not provide any information pertaining to the following:
- Training module
- Data pipeline module

## 1. Architecture

## 1.1 Solution Architecture
![image](../../assets/images/ml-pipeline.jpg)

## 1.2 Initial deployment Architecture
![image](../../assets/images/Initial_deployment_architecture.png)

## 2. Deployment package contents
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
| **logs/** | All logs generated in the project will be in this folder. They are saved as either debug.log, info.log or error.log|
|info.log|Logs events that occurred in general|
|error.log|Logs events with errors encountered|

### 2.4. Docker image

The docker image will be provided as a .tar.gz archive. The archive name is "bipo_inference_initial.tar"

| Path | Description |
| :- | - |
| **docker/** | All docker archive files will be in this folder.  Containerised image are exported into .tar.gz archive files and stored here for loading into AWS ECR service|

### 2.5. Trained model file(s)

The trained model file would be stored in AWS S3 bucket. To ensure successful download, necessary S3 permissions are required to be configured in AWS Roles settings by the AWS administrator so as to enable download of trained model files. 

| Path | Description |
| :- | - |
| **models/** | All trained models trained/saved as pickle format used in the project will be in this folder, with .pkl as extension. |
|orderedmodel_20230818.pkl|Trained model for inferencing|

Current model support:
- *statsmodels' OrderedModel* library. 

### 2.6. Key OS ports purpose.

Any firewall configurations or additional applications installation should not be using the stated ports

| Port | Description |
| :- | - |
| 2375 | Docker unencrypted communication |
| 2376 | Docker encrypted communication |
| 8000 | FastAPI endpoint|

## 3. Project File Structure in container image

The following file structure depicts the key folders/files contained in the containerised inference pipeline. **This is not the same directory with the host computer which docker is running.**

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

