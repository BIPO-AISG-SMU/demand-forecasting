#!/bin/bash

set -x

source ~/.bashrc >/dev/null

if [ ! -d "$PRED_MODEL_PATH" ]; then
    aws s3 cp $PRED_MODEL_ECS_S3_URI $HOME/from-ecs --recursive --endpoint-url="https://necs.nus.edu.sg"
fi

streamlit run src/streamlit.py -- inference.model_path=$PRED_MODEL_PATH
