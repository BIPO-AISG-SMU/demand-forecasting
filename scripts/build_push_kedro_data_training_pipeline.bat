echo off
echo "Please ensure that you have created an conda environment named bipo-df in order to polyaxon port-forwarding."
echo "Activating bipo-df env"

@REM yaml variables
set docker_registry=registry.aisingapore.net/100e-bipo
set image_name=bipo-training-pipeline:0.1.3
set docker_file=bipo-model-training-cpu.Dockerfile
echo "Activate bipo-df env"

@REM Use conda to activate bipo-df
call activate bipo-df

echo "Building image using aiap provided Dockerfile without cache"
docker build -t %docker_registry%/%image_name% -f docker/%docker_file% --platform linux/amd64 --no-cache .

if %ERRORLEVEL% equ 0 (
    echo "Pushing image to $docker_registry repo..."
    docker push %docker_registry%/%image_name%
)
PAUSE
