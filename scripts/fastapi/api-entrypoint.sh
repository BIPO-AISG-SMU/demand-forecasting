#!/bin/bash

set -e

# Set pythonpath for module import
export PYTHONPATH="$(pwd)/src"

# Start gunicorn server
gunicorn -k uvicorn.workers.UvicornWorker src.bipo_fastapi.main:app -b 127.0.0.1:8000 --chdir $(pwd)