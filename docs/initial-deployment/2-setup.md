# BIPO Demand Forecasting module: Environment Setup

## 1. Overview

This document is intended for users seeking information on the installation and configuration of the BIPO Demand Forecasting Submodule solution.

## 2. Pre-requisites

This deployment has been tested in a STAGING environment using an on-premise server based on the following specifications that mirrors the actual PRODUCTION environment to the best effort possible. We have not tested on setups which differ from what we have and unable to guarantee that the code will run without issues.

### 2.1. System specifications

 - OS: Ubuntu 22.04 LTS (Jammy Jellyfish)
 - CPU: 1 vCPU
 - RAM: 2GB
 - User: Non-root user

### 2.2. Installed OS binaries versions 

The following binaries are installed (with its dependencies) to facilitate the execution of simple network troubleshooting and docker daemon execution. Installation are done via the following command with superuser rights.

```
$ sudo apt-get install
```

|Binaries|Version|
| - | - |
|net-tools|1.60+git20181103.0eebece-1ubuntu5|
|curl|7.81.0-1ubuntu1.13|
|traceroute|1:2.1.0-2|
|apt-show-versions|0.22.13|
|ca-certificates|20230311ubuntu0.22.04.1|
|gnupg|20230311ubuntu0.22.04.1|
|docker-ce|5:24.0.5-1-ubuntu.22.04-jammy|
|docker-ce-cli|5:24.0.5-1-ubuntu.22.04-jammy|
|containerd.io|1.6.22-1|
|docker-buildx-plugin|0.11.2-1-ubuntu.22.04-jammy|
|docker-compose-plugin|2.20.2-1-ubuntu.22.04-jammy|

### 2.3 Python dependencies

All python library dependencies are install via requirements.txt in the provided docker container. No dependencies will be installed on the host as the application is already dockerised.

## 3. Preliminaries

Please extract the .zip folder provided and ensure that the name of the folder is 'bipo_demand_forecasting'. Do check the subdirectories (conf, data, logs and models) are available as they would be mounted as volume to docker directory.
```
├── bipo_demand_forecasting/
    ├── conf/ (Created and to be bind mounted)11
    ├── data/ (Created and to be bind mounted)
    ├── logs/ (Created and to be bind mounted)
    ├── models/ (Created and to be bind mounted)

Ensure that models in .pkl file are placed in models/ directory prior to mounting the directories to the containers.
```
## 4. Getting Started

### 4.1. Checking Docker service

In the following order: 

To get started, please ensure that the docker daemon is running. A quick check can be done with either of the following command:
``` bash
$ systemctl status docker
$ service docker status
```

It should show a output similar to below indicating active (running) status:
``` bash
● docker.service - Docker Application Container Engine
     Loaded: loaded (/lib/systemd/system/docker.service; enabled; vendor preset: enabled)
     Active: active (running) since Thu 2023-08-10 14:38:51 +08; 5 days ago
TriggeredBy: ● docker.socket
       Docs: https://docs.docker.com
   Main PID: 34819 (dockerd)
      Tasks: 9
     Memory: 63.9M
        CPU: 1min 11.443s
     CGroup: /system.slice/docker.service
             └─34819 /usr/bin/dockerd -H fd:// --containerd=/run/containerd/containerd.sock
```

Otherwise, please execute the following commands to restart/start the docker daemon in the VM.

``` bash
$ sudo systemctl stop docker
$ sudo systemctl start docker
```


If typing docker related command results in *"Permission denied"* error in Docker, please request your administrator to add the current login user to docker group with the following command to elevate your rights to access docker:

``` bash
$ sudo usermod -aG docker $USER
```

Docker load the provided tar.gz archive file with the following command:
``` bash
$ docker load --input bipo_demand_forecasting/docker/bipo_inference_initial.tar
```

After loading the image, check if it is loaded in the VM docker registry with the following command 'docker image ls' and you should see the following
```
$ docker image ls
REPOSITORY        TAG       IMAGE ID            CREATED       SIZE
bipo_inference    initial   <ID of the image>   XX days ago   XXXMB
```

### 4.2. Starting and Stopping Docker containers

Please refer to [chapter 3](./3-inference-module.md) and [chapter 4](./4-endpoint.md) for more detailed config.