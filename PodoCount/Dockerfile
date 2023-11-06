# This Dockerfile is used to generate the docker image dsarchive/histomicstk
# This docker image includes the HistomicsTK python package along with its
# dependencies.
#
# All plugins of HistomicsTK should derive from this docker image


# start from nvidia/cuda 10.0
# FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04
#FROM nvidia/cuda:10.2-base-ubuntu18.04
#FROM nvidia/cuda:11.1.1-base-ubuntu18.04
FROM rocker/r-ubuntu:20.04
# LABEL com.nvidia.volumes.needed="nvidia_driver"
#FROM tensorflow/tensorflow:1.15.4-gpu-py3
LABEL com.nvidia.volumes.needed="nvidia_driver"

LABEL maintainer="Sayat Mimar - Sarder Lab. <sayat.mimar@ufl.edu>"

CMD echo !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! STARTING THE BUILD !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# RUN mkdir /usr/local/nvidia && ln -s /usr/local/cuda-10.0/compat /usr/local/nvidia/lib

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

# Remove bad repos
# RUN rm \
#     /etc/apt/sources.list.d/cuda.list

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
    #python-qt4 \
    #python3-pyqt4 \
    curl \
    ca-certificates \
    libcurl4-openssl-dev \
    libexpat1-dev \
    unzip \
    libhdf5-dev \
    #libpython-dev \
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
    # needed for supporting CUDA \
    # libcupti-dev \
    # useful later \
    libmemcached-dev && \
    #apt-get autoremove && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

CMD echo !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! CHECKPOINT !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

RUN apt-get update ##[edited]
RUN apt-get install 'ffmpeg'\
    'libsm6'\
    'libxext6'  -y

RUN apt-get install libxml2-dev libxslt1-dev -y
# RUN apt-get install software-properties-common -y
# RUN add-apt-repository ppa:graphics-drivers/ppa -y
# RUN apt-get update -y
# RUN apt-get upgrade -y
# RUN apt-get install nvidia-driver-455 -y

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

# RUN wget -qO- https://cloud.r-project.org/bin/linux/ubuntu/marutter_pubkey.asc |  gpg --dearmor -o /usr/share/keyrings/r-project.gpg && \
#     echo "deb [signed-by=/usr/share/keyrings/r-project.gpg] https://cloud.r-project.org/bin/linux/ubuntu jammy-cran40/" | tee -a /etc/apt/sources.list.d/r-project.list && \
#     apt update && \
#     apt install -y r-base-dev r-recommended r-base-core r-base
# RUN apt update && \
#     apt install software-properties-common dirmngr && \
#     wget -qO- https://cloud.r-project.org/bin/linux/ubuntu/marutter_pubkey.asc |  tee -a /etc/apt/trusted.gpg.d/cran_ubuntu_key.asc && \
#     add-apt-repository "deb https://cloud.r-project.org/bin/linux/ubuntu $(lsb_release -cs)-cran40/" && \
#     apt update && \
#     apt install  -y  r-base 
# RUN apt install -y software-properties-common libblas3 liblapack3 libtcl8.6 libtk8.6 && \
#     add-apt-repository -y --enable-source --yes "ppa:marutter/rrutter4.0" && \
#     add-apt-repository -y --enable-source --yes "ppa:c2d4u.team/c2d4u4.0+" && \
#     apt install -y r-base


RUN which  python && \
    python --version
# RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
# RUN curl -O https://bootstrap.pypa.io/pip/3.7/get-pip.py && \
#     python get-pip.py && \
#     rm get-pip.py
#RUN R -e 'install.packages("Seurat",repos ="https://cran.r-project.org")'
#RUN R -e 'library(Seurat)'
ENV build_path=$PWD/build
ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
# IFTASegmentation sepcific

# copy IFTASegmentation files
ENV podo_path=$PWD/podo
RUN mkdir -p $podo_path

RUN apt-get update && \
    apt-get install -y --no-install-recommends memcached && \
    #apt-get autoremove && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
# RUN pip install torch
# RUN python -c 'import torch,sys;print(torch.cuda.is_available());sys.exit(not torch.cuda.is_available())'
COPY . $podo_path/
WORKDIR $podo_path

#   Upgrade setuptools, as the version in Conda won't upgrade cleanly unless it
# is ignored.


RUN pip install --no-cache-dir --upgrade --ignore-installed pip setuptools && \
    pip install --no-cache-dir .  && \
    #pip install --no-cache-dir 'tensorflow==1.15' && \
    # Install large_image memcached extras \
    #pip install --no-cache-dir 'large-image[memcached]' && \
    # Install HistomicsTK \
    #pip install --no-cache-dir . --find-links https://girder.github.io/large_image_wheels && \
    # Install tf-slim \
    #pip install --no-cache-dir 'tf-slim>=1.1.0' && \
    # Install pillow_lut \
    #pip install --no-cache-dir 'pillow-lut' && \

    #pip install --no-cache-dir tensorboard cmake onnx && \

    #pip install --no-cache-dir torch==1.10  torchaudio==0.10 torchvision==0.11.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html && \

    #pip install --no-cache-dir 'git+https://github.com/facebookresearch/fvcore' && \

    #git clone https://github.com/facebookresearch/detectron2 detectron2_repo && \
    #git clone https://github.com/facebookresearch/detectron2.git && \
    #python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.10/index.html && \
    #python -m pip install -e detectron2 && \
    # clean up \
    rm -rf /root/.cache/pip/*

# ENV FORCE_CUDA="1"
# ARG TORCH_CUDA_ARCH_LIST="Kepler;Kepler+Tesla;Maxwell;Maxwell+Tegra;Pascal;Volta;Turing"
# ENV TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}"
# RUN pip install --no-cache-dir -e detectron2_repo
# Show what was installed
RUN python --version && pip --version && pip freeze

#RUN pip install --user torch==1.10 torchvision==0.11.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html

#RUN pip install --user 'git+https://github.com/facebookresearch/fvcore'
# install detectron2
#RUN git clone https://github.com/facebookresearch/detectron2 detectron2_repo
# set FORCE_CUDA because during `docker build` cuda is not accessible
#ENV FORCE_CUDA="1"
# remove cuda compat
# RUN apt remove --purge cuda-compat-10-0 --yes

# pregenerate font cache
RUN python -c "from matplotlib import pylab"

# Suppress warnings
# RUN sed -i 's/^_PRINT_DEPRECATION_WARNINGS = True/_PRINT_DEPRECATION_WARNINGS = False/g' /usr/local/lib/python3.8/dist-packages/tensorflow_core/python/util/deprecation.py && \
#     sed -i 's/rename = get_rename_v2(full_name)/rename = False/g' /usr/local/lib/python3.8/dist-packages/tensorflow_core/python/util/module_wrapper.py

# define entrypoint through which all CLIs can be run
WORKDIR $podo_path/podo/cli

# Test our entrypoint.  If we have incompatible versions of numpy and
# openslide, one of these will fail
RUN python -m slicer_cli_web.cli_list_entrypoint --list_cli
RUN python -m slicer_cli_web.cli_list_entrypoint PodoCount --help


ENTRYPOINT ["/bin/bash", "docker-entrypoint.sh"]