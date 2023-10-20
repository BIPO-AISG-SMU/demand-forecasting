# Deployment Environment Setup - Training Module

## Overview
This document is intended for users seeking information on the installation and configuration of the BIPO Demand Forecasting Module.

## Pre-requisites

This deployment has been tested in a **staging** environment using an on-premise server based on the following specifications that mirrors the actual/production environment to the best effort possible. We have not tested on set-ups which differ from what we have and unable to guarantee that the code will run without issues.

### System Specifications

 - OS: `Windows 10 Pro with WSL2 installed`
 - CPU: `4 vCPU`
 - RAM: `16GB`
 - User: Administrator

## Installation of dependencies in Windows
| Software | Purpose |
| --- | --- |
| [Docker Desktop](https://docs.docker.com/desktop/install/windows-install/) | For running training pipeline image application. (Mandatory) |
| [Miniconda](https://docs.conda.io/projects/miniconda/en/latest/miniconda-install.html) | Helps you create an environment for many different versions of Python and package versions. (Optional) |

### Python Dependencies
All Python library dependencies are install via `requirements.txt` within the provided Docker container. No dependencies will be installed on the host as the application is containerised.

## Getting Started
This assumes that the dependencies has been installed in part 1.
1. Unzip the final deployment package manually or use the commend below in powershell: 
~~~
Expand-Archive -Path "100E_BIPO_final_deployment_20231019.zip" -DestinationPath "100E_BIPO_final_deployment_20231019" -Force
~~~
2. After unzipping, do ensure that the subdirectories below are present as they would be mounted as volume to the Docker container.

```
├──/app/bipo_demand_forecasting/
    ├── scripts (not for mounting)
    |   └── run_model_training.bat
    ├── docs (not for mounting)
    ├── docker (not for mounting)
    ├── conf/ (to be bind mounted)
    |   ├── base/ 
    |   |   ├──catalog.yml
    |   |   ├──constants.yml
    |   |   ├──inference_parameters.yml
    |   |   ├──logging.yml
    |   |   └──parameters.yml
    |   ├── __init__.py
    |   ├── parameters/ 
    |   |   ├── data_split.yml
    |   |   └─ model_training.yml
    |   └─ local/ (empty folder)
    |      └─ ...
    ├── data/ (to be bind mounted)
    ├── mlruns/ (to be bind mounted)
    |   └── 0
    |       └── metal.yaml 
    ├── models/ (to be bind mounted)
    |   ├── ebm_model.pkl/
    |   |   └──/2023-10-19T02.22.52.045Z/
    |   |      └── ebm_model.pkl (Model file)
    |   └── ordered_model.pkl
    |       └──/2023-10-19T02.40.38.841Z/
    |          └── ordered_model.pkl (Model file)
    └── logs/ (to be bind mounted)

```
3. Run Docker Desktop application.
4. Load the provided tar archive file using the following command:
~~~
docker load --input .\bipo_demand_forecasting\docker\100E_BIPO_docker_training.tar
~~~

After loading the image, check if it is loaded in the VM. You can view image on Docker Desktop or with the `docker image ls` command. You should see the following:

~~~powershell
# run command in powershell
docker image ls

# output
REPOSITORY                                                   TAG       IMAGE ID            CREATED       SIZE
registry.aisingapore.net/100e-bipo/bipo-training-pipeline    initial   <ID of the image>   XX days ago   XXXMB
~~~

## Running Training Pipeline in Docker Container
Please refer to [training pipeline](training-pipeline) for more details.

## Setting up MLflow on the Local Machine (Optional)
If you want to use MLflow for tracking ML experiments, follow these steps:

1. Create and activate conda environment with Python:
```python
conda create -n mlflow-env python=3.10
conda activate mlflow-env
```  
2. Install MLflow in conda environment using pip:
```python
pip install mlflow
```  
3. Verify MLflow is installed successfully:
```python
mlflow --version

# example output:
# mlflow, version 2.7.1
```  
Now you can view MLflow web UI on your browser. 