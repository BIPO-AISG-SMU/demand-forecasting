FROM debian:11

ARG NON_ROOT_USER="coder"
ARG HOME_DIR="/home/$NON_ROOT_USER"
# VSCode Server arguments
ARG CODE_SERVER_VERSION="4.0.1"
ARG CODE_SERVER_BINARY_NAME="code-server_4.0.1_amd64.deb"
# Miniconda arguments
ARG CONDA_HOME="/miniconda3"
ARG CONDA_BIN="$CONDA_HOME/bin/conda"
ARG MINI_CONDA_SH="Miniconda3-py39_4.12.0-Linux-x86_64.sh"

RUN apt-get update \
    && apt-get install -y \
    curl \
    dumb-init \
    zsh \
    htop \
    locales \
    man \
    nano \
    git \
    procps \
    openssh-client \
    sudo \
    vim.tiny \
    lsb-release \
    wget \
    unzip \
    apt-transport-https ca-certificates gnupg && \
    rm -rf /var/lib/apt/lists/*

# Install AWS CLI v2
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" && \
    unzip awscliv2.zip && \
    ./aws/install

# RUN apt-get update && \
#     apt-get install -y \
#     kubectl
RUN mkdir -p /etc/apt/keyrings && \
    curl -fsSL "https://packages.cloud.google.com/apt/doc/apt-key.gpg" | gpg --dearmor -o /etc/apt/trusted.gpg.d/kubernetes-archive-keyring.gpg && \
    echo 'deb https://packages.cloud.google.com/apt kubernetes-xenial main' > /etc/apt/sources.list.d/kubernetes.list && \
    sudo apt-get update && \
    sudo apt-get install -y kubectl

RUN curl https://baltocdn.com/helm/signing.asc | sudo apt-key add - && \
    echo "deb https://baltocdn.com/helm/stable/debian/ all main" | sudo tee /etc/apt/sources.list.d/helm-stable-debian.list && \
    apt-get update && \
    apt-get install -y helm

RUN wget https://github.com/mikefarah/yq/releases/download/v4.16.1/yq_linux_amd64.tar.gz -O - |\
    tar xz && mv yq_linux_amd64 /usr/bin/yq

# https://wiki.debian.org/Locale#Manually
RUN sed -i "s/# en_US.UTF-8/en_US.UTF-8/" /etc/locale.gen \
    && locale-gen
ENV LANG=en_US.UTF-8

RUN adduser --gecos '' --disabled-password coder && \
    echo "coder ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers.d/nopasswd

RUN ARCH="$(dpkg --print-architecture)" && \
    curl -fsSL "https://github.com/boxboat/fixuid/releases/download/v0.5/fixuid-0.5-linux-$ARCH.tar.gz" | tar -C /usr/local/bin -xzf - && \
    chown root:root /usr/local/bin/fixuid && \
    chmod 4755 /usr/local/bin/fixuid && \
    mkdir -p /etc/fixuid && \
    printf "user: coder\ngroup: coder\n" > /etc/fixuid/config.yml

COPY aisg-context/code-server/entrypoint.sh /usr/bin/entrypoint.sh

RUN wget "https://github.com/cdr/code-server/releases/download/v$CODE_SERVER_VERSION/$CODE_SERVER_BINARY_NAME"

RUN chown -R 2222:2222 $CODE_SERVER_BINARY_NAME && \
    chown -R 2222:2222 /usr/bin/entrypoint.sh && \
    chmod +x /usr/bin/entrypoint.sh

RUN dpkg -i $CODE_SERVER_BINARY_NAME && rm $CODE_SERVER_BINARY_NAME

RUN mkdir $CONDA_HOME && chown -R 2222:2222 $CONDA_HOME
RUN chown -R 2222:2222 $HOME_DIR

EXPOSE 8080
# This way, if someone sets $DOCKER_USER, docker-exec will still work as
# the uid will remain the same. note: only relevant if -u isn't passed to
# docker-run.
USER 2222
ENV USER=$NON_ROOT_USER
WORKDIR $HOME_DIR

# Install Miniconda
RUN curl -O "https://repo.anaconda.com/miniconda/$MINI_CONDA_SH" && \
    chmod +x $MINI_CONDA_SH && \
    ./$MINI_CONDA_SH -u -b -p $CONDA_HOME && \
    rm $MINI_CONDA_SH
ENV PATH $CONDA_HOME/bin:$HOME_DIR/.local/bin:$PATH

ENTRYPOINT ["/usr/bin/entrypoint.sh", "--bind-addr", "0.0.0.0:8080", "."]
