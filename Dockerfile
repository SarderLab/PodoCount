FROM rocker/r-ubuntu:20.04

LABEL com.nvidia.volumes.needed="nvidia_driver"

LABEL maintainer="Sayat Mimar - Sarder Lab. <sayat.mimar@ufl.edu>"

CMD echo !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! STARTING THE BUILD !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

RUN apt-get update && \
    apt-get install --yes --no-install-recommends software-properties-common && \
    # As of 2018-04-16 this repo has the latest release of Python 2.7 (2.7.14) \
    # add-apt-repository ppa:jonathonf/python-2.7 && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get --yes --no-install-recommends -o Dpkg::Options::="--force-confdef" -o Dpkg::Options::="--force-confold" dist-upgrade && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    #keyboard-configuration \
    git \
    wget \
    curl \
    ca-certificates \
    libcurl4-openssl-dev \
    libexpat1-dev \
    unzip \
    libhdf5-dev \
    libpython3-dev \
    python2.7-dev \
    python-tk \
    # We can't go higher than 3.7 and use tensorflow 1.x \
    python3.8-dev \
    python3.8-distutils \
    python3-tk \
    software-properties-common \
    libssl-dev \
    # Standard build tools \
    build-essential \
    cmake \
    autoconf \
    automake \
    libtool \
    pkg-config \
    libmemcached-dev && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

CMD echo !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! CHECKPOINT !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

RUN apt-get update ##[edited]
RUN apt-get install 'ffmpeg'\
    'libsm6'\
    'libxext6'  -y

RUN apt-get install libxml2-dev libxslt1-dev -y

WORKDIR /
# Make Python3 the default and install pip.  Whichever is done last determines
# the default python version for pip.

#Make a specific version of python the default and install pip
RUN rm -f /usr/bin/python && \
    rm -f /usr/bin/python3 && \
    ln `which python3.8` /usr/bin/python && \
    ln `which python3.8` /usr/bin/python3 && \
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python get-pip.py && \
    rm get-pip.py && \
    ln `which pip3` /usr/bin/pip 

RUN which  python && \
    python --version

ENV build_path=$PWD/build
ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

ENV podo_path=$PWD/podo
RUN mkdir -p $podo_path

RUN apt-get update && \
    apt-get install -y --no-install-recommends memcached && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

COPY . $podo_path/
WORKDIR $podo_path

# Upgrade setuptools, as the version in Conda won't upgrade cleanly unless it
# is ignored.

RUN pip install --no-cache-dir --upgrade --ignore-installed pip setuptools && \
    pip install --no-cache-dir .  && \
    rm -rf /root/.cache/pip/*

# Show what was installed
RUN python --version && pip --version && pip freeze

# Define entrypoint through which all CLIs can be run
WORKDIR $podo_path/podo/cli

# Test our entrypoint.  If we have incompatible versions of numpy and
# Openslide, one of these will fail
RUN python -m slicer_cli_web.cli_list_entrypoint --list_cli
RUN python -m slicer_cli_web.cli_list_entrypoint PodoCount --help

ENTRYPOINT ["/bin/bash", "docker-entrypoint.sh"]