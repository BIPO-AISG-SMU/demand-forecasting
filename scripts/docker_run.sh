#!/bin/sh

# SYSTEM VARIABLES
PROJECT_FOLDER=bipo_demand_forecasting
docker_image_name=registry.aisingapore.net/100e-bipo/bipo_inference:initial
container_user_path=/app
echo "Switching to $HOME directory"
cd ~

#echo "Repulling image for latest changes"
#docker pull $docker_image_name

echo "Loading docker tar file"
docker load -i $HOME/$PROJECT_FOLDER/docker/bipo_inference.tar

echo "Running docker image on port 8000 with port forwarding"
docker run -p 8000:8000\
        --name bipo_inference_initial \
        --mount type=bind,source=$HOME/$PROJECT_FOLDER/conf,target=$container_user_path/$PROJECT_FOLDER/conf \
        --mount type=bind,source=$HOME/$PROJECT_FOLDER/data,target=$container_user_path/$PROJECT_FOLDER/data \
        --mount type=bind,source=$HOME/$PROJECT_FOLDER/models,target=$container_user_path/$PROJECT_FOLDER/models \
        --mount type=bind,source=$HOME/$PROJECT_FOLDER/logs,target=$container_user_path/$PROJECT_FOLDER/logs \
        --rm \
        $docker_image_name