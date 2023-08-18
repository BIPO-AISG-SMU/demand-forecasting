# Environment Setup

## 1. Overview

This document is intended for users seeking information on the installation and configuration of the BIPO Demand Forecasting Module.

## 2. Pre-requisites

This deployment has been tested in a **staging** environment using an on-premise server based on the following specifications that mirrors the actual/production environment to the best effort possible. We have not tested on set-ups which differ from what we have and unable to guarantee that the code will run without issues.

### 2.1. System Specifications

 - OS: Ubuntu 22.04 LTS (Jammy Jellyfish)
 - CPU: 1 vCPU
 - RAM: 2GB
 - User: Non-root user

### 2.2. Installation 

The following steps for installation assumes that the VM has internet access. 
Unpack and extract the zip folder provided. 

```
unzip bipo_inference
```


#### 2.2.1 Installing from script

**NOTE**
1. Before installing the necessary binaries, please ensure that the USER variable in apt-install.sh located in bipo_demand_forecasting/scripts/apt-install.sh is set to the correct VM user. Otherwise, wrong ownership will be set to the folders

Open a terminal and navigate to your $HOME folder (i.e /home/<username>). Please run the following command
```
$ sudo bash bipo_demand_forecasting/src/apt-install.sh
```

The contents of the `apt-install.sh` script are as follows:
```
#!/usr/bin/sh
set -e
#set -x

# Bash Variables. Bipo main directory and its subdir folder
bipo_dir=bipo_demand_forecasting
USER=aisg # Please amend based on your VM username
cd "$(dirname "${BASH_SOURCE[0]}")"

# Install necessary binaries for debugging and troubleshooting
echo "Installing basic binaries such as net-tools, curl, traceroute and apt-show-versions"
apt-get install -y net-tools traceroute apt-show-versions

echo "Setting up installation process for docker engine in Ubuntu..."
#Update the apt package index and install packages to allow apt to use a repository over HTTPS:
apt-get install -y ca-certificates curl gnupg

# Add Docker offical GPG key
echo "Adding Docker GPG key"
install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
chmod a+r /etc/apt/keyrings/docker.gpg

# Setup repository
echo \
        "deb [arch="$(dpkg --print-architecture)" signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
        "$(. /etc/os-release && echo "$VERSION_CODENAME")" stable" | \
        tee /etc/apt/sources.list.d/docker.list > /dev/null

# update the package index files on the Ubuntu system
apt-get update

# Install docker binaries and dependencies
echo "Installing docker binaries and dependencies..."
apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

echo "Downloading AWS CLI v2 installer"
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"

echo "Unzipping and installing AWS CLI v2 installer"
unzip awscliv2.zip && ./aws/install

echo "Cleaning up awscliv2.zip folders"
rm -rf awscliv2.zip*

# Make bipo directory and its subdirectory
echo "Creating directories for $bipo_dir if it does not exist"
if ! [ -d $bipo_dir ]
then
        mkdir $bipo_dir && cd $bipo_dir
        echo "Creating conf data logs and docker subdirectory"
        mkdir conf data logs docker
        echo "Moving out of directory"
        cd ..
else
        echo "$bipo_dir exists.Creating necessary subfolders if required."
        cd $bipo_dir && mkdir -p conf data logs docker && cd ..
fi
echo "Changing owner:group to $USER with rwxr-xr-x permissions"
chown -R $USER:$USER $bipo_dir/ && chmod 755 -R $bipo_dir/
```

#### 2.2.2 List of installed binaries and versions
Upon installation, you would have the following binaries installed.

|Binary|Version|
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

### 2.3. Python Dependencies

All Python library dependencies are install via `requirements.txt` within the provided docker container. No dependencies will be installed on the host as the application is containerised.

## 3. Preliminary Steps

1. Extract the .zip folder provided and ensure that the name of the extracted directory is `bipo_demand_forecasting`. Do check that the below subdirectories are present as they would be mounted as volume to the Docker container.
```
├── bipo_demand_forecasting/
    ├── conf/ (Created and to be bind mounted)
    ├── data/ (Created and to be bind mounted)
    ├── logs/ (Created and to be bind mounted)
    ├── models/ (Created and to be bind mounted)

```
2. Ensure that the trained model(s) (in `.pkl` format) are placed in the `models` subdirectory prior to mounting the directory to the container.

## 4. Getting Started

### 4.1. Checking Docker Service

In the following order: 

1. To get started, please ensure that the Docker daemon is running. A quick check can be done with either of the following commands:
``` bash
$ systemctl status docker
$ service docker status
```

2. It should show an output similar to the below, indicating active (running) status:
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

- Otherwise, please execute the following commands to restart/start the Docker daemon in the VM:

``` bash
$ sudo systemctl stop docker
$ sudo systemctl start docker
```

3. Docker load the provided tar.gz archive file using the following command:
``` bash
$ docker load --input bipo_demand_forecasting/docker/bipo_inference_initial.tar
```

4. After loading the image, check if it is loaded in the VM Docker registry with the command `docker image ls` and you should see the following:
```
$ docker image ls
REPOSITORY        TAG       IMAGE ID            CREATED       SIZE
bipo_inference    initial   <ID of the image>   XX days ago   XXXMB
```

If typing Docker commands results in a *"Permission denied"* error, please request your administrator to add the current user to Docker group with the following command to elevate your rights to access Docker:

``` bash
$ sudo usermod -aG docker $USER
```

### 4.2. Starting and Stopping Docker Containers

Please refer to [Chapter 3: Inference Module](./3-inference-module.md) and [Chapter 4: Model Endpoint](./4-endpoint.md) for more details.

### 4.3 Lifting of firewall ports in VM (if needed)
Please execute the following to allow any incoming connections to port 8080 with the following command. 
Please note that this requires sudo privilege. 
```
sudo ufw allow from any to any port 8080 proto tcp
```