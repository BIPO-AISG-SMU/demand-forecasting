echo off

@REM Define your Docker image and registry
set docker_registry=registry.aisingapore.net/100e-bipo
set image_name=bipo-training-pipeline:0.1.3
set image_tag_name=%docker_registry%/%image_name%

@REM Define volume variables from host directories
set DATA_DIR=data
set CONF_DIR=conf
set MODELS_DIR=models
set LOGS_DIR=logs
set MLRUN_DIR=mlruns

echo "Pulling image from $docker_registry repo..."
@REM If docker_registry is empty, skip docker pull
if not "%docker_registry%"=="" (
    echo "Pulling image from %docker_registry% repo..."
    docker pull %docker_registry%/%image_name%
    echo "Successfully pull image: %image_tag_name%"
) else (
    echo "Docker registry not provided. Skipping image pull."
    @REM If docker_registry is empty, docker run based on image name
    set image_tag_name=%image_name%
)

@REM Run docker image with the mounted volumes and execute kedro run
@REM rm to remove the container after it has executed successfully
@REM -i: Keeps STDIN open, allowing you to interact with the container.
@REM -t: Allocates a pseudo-TTY for the container, which can make the interaction more user-friendly.
@REM -d: Runs the container in detached mode, meaning it runs in the background and doesn't occupy your terminal.
@REM sh -c shell commend to run kedro after container is running. Only training pipeline is run here.
@REM mlflow data will be saved in mlruns

echo "Starting docker run with image: %image_tag_name% "
docker run --rm ^
-v "%CD%\%CONF_DIR%:/home/kedro_docker/%CONF_DIR%" ^
-v "%CD%\%DATA_DIR%:/home/kedro_docker/%DATA_DIR%" ^
-v "%CD%\%MODELS_DIR%:/home/kedro_docker/%MODELS_DIR%" ^
-v "%CD%\%LOGS_DIR%:/home/kedro_docker/%LOGS_DIR%" ^
-v "%CD%\%MLRUN_DIR%:/home/kedro_docker/%MLRUN_DIR%" ^
-itd %image_tag_name% ^
sh -c "kedro run --pipeline=training_pipeline"

echo "Training Pipeline Docker Container run successfully"

@REM Check if mlflow is installed
where mlflow > nul 2>&1
if %errorlevel% neq 0 (
    echo "mlflow is not installed on your local environment. Please install it to view mlflow."
    exit /b 1
)

@REM mlflow is installed, so run mlflow ui
echo "mlflow is installed on your local environment. Running mlflow"
mlflow ui