################################################################################
# SPDX-License-Identifier: Apache-2.0                                          #
#                                                                              #
# Copyright (C) 2021, Arm Limited and contributors                             #
#                                                                              #
# Licensed under the Apache License, Version 2.0 (the "License");              #
# you may not use this file except in compliance with the License.             #
# You may obtain a copy of the License at                                      #
#                                                                              #
#     http://www.apache.org/licenses/LICENSE-2.0                               #
#                                                                              #
# Unless required by applicable law or agreed to in writing, software          #
# distributed under the License is distributed on an "AS IS" BASIS,            #
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.     #
# See the License for the specific language governing permissions and          #
# limitations under the License.                                               #
################################################################################
#
# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

ARG UBUNTU_VERSION=20.04

ARG ARCH=arm64
ARG CUDA=11.3
FROM nvidia/cuda${ARCH:+-$ARCH}:${CUDA}.1-base-ubuntu${UBUNTU_VERSION} as base
# ARCH and CUDA are specified again because the FROM directive resets ARGs
# (but their default value is retained if set previously)
ARG ARCH
ARG CUDA
ARG CUDNN=8.2.1.32-1
ARG CUDNN_MAJOR_VERSION=8
ARG LIB_DIR_PREFIX=aarch64
ARG LIBNVINFER=8.0.1-1
ARG LIBNVINFER_MAJOR_VERSION=8

# Needed for string substitution
SHELL ["/bin/bash", "-c"]
# Pre-install tzdata making sure it won't ask for anything midway
RUN apt-get update && DEBIAN_FRONTEND="noninteractive" TZ=Europe/London apt-get install -y tzdata
# Pick up some TF dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        cuda-command-line-tools-${CUDA/./-} \
        libcublas-${CUDA/./-} \
        libcublas-dev-${CUDA/./-} \
        cuda-nvrtc-${CUDA/./-} \
        libcufft-${CUDA/./-} \
        libcufft-dev-${CUDA/./-} \
        libcurand-${CUDA/./-} \
        libcurand-dev-${CUDA/./-} \
        libcusolver-${CUDA/./-} \
        libcusolver-dev-${CUDA/./-} \
        libcusparse-${CUDA/./-} \
        libcusparse-dev-${CUDA/./-} \
        curl \
        libcudnn8=${CUDNN}+cuda${CUDA} \
        libcudnn8-dev=${CUDNN}+cuda${CUDA} \
        libfreetype6-dev \
        libhdf5-serial-dev \
        libzmq3-dev \
        pkg-config \
        software-properties-common \
        unzip

# Install TensorRT if not building for PowerPC
# NOTE: libnvinfer uses cuda${CUDA} versions
RUN [[ "${ARCH}" = "ppc64le" ]] || { apt-get update && \
        apt-get install -y --no-install-recommends libnvinfer${LIBNVINFER_MAJOR_VERSION}=${LIBNVINFER}+cuda${CUDA} \
        libnvinfer-plugin${LIBNVINFER_MAJOR_VERSION}=${LIBNVINFER}+cuda${CUDA} \
        && apt-get clean \
        && rm -rf /var/lib/apt/lists/*; }

# For CUDA profiling, TensorFlow requires CUPTI.
ENV LD_LIBRARY_PATH /usr/local/cuda/extras/CUPTI/lib64:/usr/local/cuda/lib64:/usr/local/cuda-${CUDA}/lib64:$LD_LIBRARY_PATH

# Link the libcuda stub to the location where tensorflow is searching for it and reconfigure
# dynamic linker run-time bindings
RUN ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1 \
    && echo "/usr/local/cuda/lib64/stubs" > /etc/ld.so.conf.d/z-cuda-stubs.conf \
    && ldconfig

# See http://bugs.python.org/issue19846
ENV LANG C.UTF-8

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip

RUN python3 -m pip --no-cache-dir install --upgrade \
    "pip<20.3" \
    setuptools

# Some TF tools expect a "python" binary
RUN ln -s $(which python3) /usr/local/bin/python

# Install bazel
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    wget \
    openjdk-8-jdk \
    python3-dev \
    virtualenv \
    swig

RUN python3 -m pip --no-cache-dir install \
    Pillow \
    h5py \
    keras_preprocessing \
    matplotlib \
    mock \
    'numpy<1.19.0' \
    scipy \
    sklearn \
    pandas \
    future \
    portpicker \
    seaborn \
    enum34

ARG BAZEL_VERSION=3.1.0
RUN wget -O /usr/local/bin/bazel "https://github.com/bazelbuild/bazel/releases/download/3.5.1/bazel-3.5.1-linux-arm64" && \
    chmod +x /usr/local/bin/bazel

# Build and install Tensorflow
ARG TF_PACKAGE=tensorflow-gpu
ARG TF_PACKAGE_VERSION=2.4.1
COPY required_files/aarch64-build-gpu.sh /build-gpu.sh
COPY required_files/tf-v2.4.1_tflite_randomUniform.patch /tflite.patch
RUN /bin/bash /build-gpu.sh

# Install TensorFlow datasets
RUN python3 -m pip --no-cache-dir install tensorflow_datasets

# Put a nice MOTD?
COPY required_files/bashrc /etc/bash.bashrc
RUN chmod a+rwx /etc/bash.bashrc
