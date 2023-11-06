# Use the official Python base image
FROM python:3.10-slim

# Build environement variables
#ARG NON_ROOT_USER=bipo_user
ARG HOME_DIR=/app #/home/$NON_ROOT_USER

# Set environment paths
ENV BIPO_PROJECT=$HOME_DIR/bipo_demand_forecasting
ENV PYTHONPATH=$HOME_DIR/bipo_demand_forecasting/src/
ENV API_NAME=BIPO_FastAPI
ENV LOGGER_CONFIG_PATH=$BIPO_PROJECT/conf/base/logging_inference.yml
ENV PRED_MODEL_UUID=0.1
ENV PRED_MODEL_PATH=$BIPO_PROJECT/models/orderedmodel_prob_20230816.pkl
ENV INTERMEDIATE_OUTPUT_PATH=$BIPO_PROJECT/data/10_model_inference_output

# Create home directory
RUN mkdir -p $BIPO_PROJECT/conf/base $BIPO_PROJECT/conf/local $BIPO_PROJECT/data $BIPO_PROJECT/models $BIPO_PROJECT/logs

# Set work directory
WORKDIR $HOME_DIR/bipo_demand_forecasting

# Copy this toml (used as dependencies by python script)
COPY . ./

# Copy src folder
#COPY src src/

# Install the Python dependencies and switch directory to src/ to start fastapi
RUN cd src/ && pip install -r requirements.txt #&& chown -R 2222:2222 $HOME_DIR 

# Set user 
#USER 2222
#ENV USER=$NON_ROOT_USER

# Expose the port on which the fastapi application will run
EXPOSE 8000

# Run the FastAPI application using uvicorn server
CMD ["uvicorn", "bipo_fastapi.main:app", "--host", "0.0.0.0", "--port", "8000"]
