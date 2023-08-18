#!/usr/bin/sh
set -e
#set -x

# Bash Variables. Bipo main directory and its subdir folder
BIPO_DIR=bipo_demand_forecasting
USER=aisg #Please amend based on your VM username

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

echo "Unzipping $BIPO_DIR in $HOME"
cd ~
unzip $BIPO_DIR.zip -d ./$BIPO_DIR


echo "Changing owner:group to $USER with rwxr-xr-x permissions"
chown -R $USER:$USER $BIPO_DIR/ && chmod 755 -R $BIPO_DIR/