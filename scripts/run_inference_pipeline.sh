#!/bin/bash

set -e

# Define your Docker image and registry
docker_registry=""
image_name="bipo-training-pipeline:0.1.5"
image_tag_name="$docker_registry/$image_name"

# Define volume variables from host directories
DATA_DIR="data"
CONF_DIR="conf"
MODELS_DIR="models"
LOGS_DIR="logs"
MLRUN_DIR="mlruns"

echo "Pulling image from $docker_registry repo..."

# If docker_registry is not empty, pull the image
if [ -n "$docker_registry" ]; then
    echo "Pulling image from $docker_registry repo..."
    docker pull "$docker_registry/$image_name"
    echo "Successfully pulled image: $image_tag_name"
else
    echo "Docker registry not provided. Skipping image pull."
    # If docker_registry is empty, use image name only
    image_tag_name="$image_name"
fi

# Run docker image with the mounted volumes and execute kedro run
# rm to remove the container after it has executed successfully
# -i: Keeps STDIN open, allowing you to interact with the container.
# -t: Allocates a pseudo-TTY for the container, which can make the interaction more user-friendly.
# -d: Runs the container in detached mode, meaning it runs in the background and doesn't occupy your terminal.
# sh -c shell command to run kedro after the container is running. Only the training pipeline is run here.
# mlflow data will be saved in mlruns

echo "Starting docker run with image: $image_tag_name"
docker run -p 8000:8000 --rm \
-v "$PWD/$CONF_DIR:/home/kedro_docker/$CONF_DIR" \
-v "$PWD/$DATA_DIR:/home/kedro_docker/$DATA_DIR" \
-v "$PWD/$MODELS_DIR:/home/kedro_docker/$MODELS_DIR" \
-v "$PWD/$LOGS_DIR:/home/kedro_docker/$LOGS_DIR" \
-v "$PWD/$MLRUN_DIR:/home/kedro_docker/$MLRUN_DIR" \
-it "$image_tag_name" \
sh -c "gunicorn -k uvicorn.workers.UvicornWorker src.bipo_fastapi.main:app -b :8000 --chdir \$(pwd)"

echo "Inference Pipeline Docker Container started successfully"