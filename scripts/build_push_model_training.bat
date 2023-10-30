echo off
echo "Please ensure that you have created an conda environment named bipo-df in order to polyaxon port-forwarding."
echo "Activating bipo-df env"

: yaml variables
set env_name=training_pipeline
set docker_registry=registry.aisingapore.net/100e-bipo
set image_name=bipo-training-pipeline-on-polyaxon:0.1.0
set docker_file=bipo-kedro-training-pipeline.Dockerfile
echo "Activate bipo-df env"

: Use conda to activate training-pipeline
call activate %env_name%

echo "Building image with uid and gid set to default..."
kedro docker build --uid=2222 --gid=1000 --image %docker_registry%/%image_name%  --docker-args "-f docker/%docker_file% --no-cache"

if %ERRORLEVEL% equ 0 (
    echo "Pushing image to %docker_registry% repo..."
    docker push %docker_registry%/%image_name%
)
PAUSE
