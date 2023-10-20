echo off

@REM Define your Docker image and registry
set docker_registry=registry.aisingapore.net/100e-bipo
set image_name=bipo-training-pipeline:0.1.3
set image_tag_name=%docker_registry%/%image_name%

@REM Define volume variables from host directories
set DATA_DIR=data
set CONF_DIR=conf
set LOGS_DIR=logs

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
@REM sh -c shell commend to run kedro after container is running. Only training pipeline is run here.
echo "Running Data Pipeline Docker Container"
docker run --rm ^
-v "%CD%\%CONF_DIR%:/home/kedro_docker/%CONF_DIR%" ^
-v "%CD%\%DATA_DIR%\:/home/kedro_docker/%DATA_DIR%" ^
-v "%CD%\%LOGS_DIR%:/home/kedro_docker/%LOGS_DIR%" ^
-itd %image_tag_name% ^
sh -c "kedro run --pipeline=data_loader && kedro run --pipeline=data_pipeline"

echo "Data Pipeline Docker Container run successfully"