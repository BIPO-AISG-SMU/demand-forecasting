#!/bin/bash

set -x

source ~/.bashrc

if [ ! -d "$PRED_MODEL_PATH" ]; then
    aws s3 cp $PRED_MODEL_ECS_S3_URI $HOME/from-ecs --recursive --endpoint-url="https://necs.nus.edu.sg"
fi

cd src
gunicorn bipo_fastapi.main:APP -b 0.0.0.0:8080 -w 4 -k uvicorn.workers.UvicornWorker --timeout 90
