FROM ubuntu:18.04

ARG REPO_DIR="."
ARG CONDA_ENV_FILE="bipo-conda-env.yml"
ARG CONDA_ENV_NAME="bipo"
ARG PROJECT_USER="aisg"
ARG HOME_DIR="/home/$PROJECT_USER"

# Miniconda arguments
ARG CONDA_HOME="/miniconda3"
ARG CONDA_BIN="$CONDA_HOME/bin/conda"
ARG MINI_CONDA_SH="Miniconda3-py39_4.12.0-Linux-x86_64.sh"

WORKDIR $HOME_DIR

RUN groupadd -g 2222 $PROJECT_USER && useradd -u 2222 -g 2222 -m $PROJECT_USER

RUN touch "$HOME_DIR/.bashrc"

RUN apt-get update && \
    apt-get -y install bzip2 curl wget gcc rsync git vim locales && \
    sed -i -e 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen && \
    locale-gen && \
    dpkg-reconfigure --frontend=noninteractive locales && \
    update-locale LANG=en_US.UTF-8 && \
    apt-get clean

RUN wget https://github.com/mikefarah/yq/releases/download/v4.16.1/yq_linux_amd64.tar.gz -O - |\
    tar xz && mv yq_linux_amd64 /usr/bin/yq

COPY $REPO_DIR bipo

RUN mkdir $CONDA_HOME && chown -R 2222:2222 $CONDA_HOME
RUN chown -R 2222:2222 $HOME_DIR && \
    rm /bin/sh && ln -s /bin/bash /bin/sh

ENV PYTHONIOENCODING utf8
ENV LANG "C.UTF-8"
ENV LC_ALL "C.UTF-8"

USER 2222

# Install Miniconda
RUN curl -O https://repo.anaconda.com/miniconda/$MINI_CONDA_SH && \
    chmod +x $MINI_CONDA_SH && \
    ./$MINI_CONDA_SH -u -b -p $CONDA_HOME && \
    rm $MINI_CONDA_SH
ENV PATH $CONDA_HOME/bin:$HOME_DIR/.local/bin:$PATH
# Install conda environment
RUN $CONDA_BIN env create -f bipo/$CONDA_ENV_FILE && \
    $CONDA_BIN init bash && \
    $CONDA_BIN clean -a -y && \
    echo "source activate $CONDA_ENV_NAME" >> "$HOME_DIR/.bashrc"
