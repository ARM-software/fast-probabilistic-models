#!/usr/bin/env bash

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
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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

# Get TensorFlow sources and apply the TFLite patch adding RandomUniform operator
set -ex
git clone --branch=v2.4.1 --depth=1 https://github.com/tensorflow/tensorflow.git /tensorflow || true
cd /tensorflow
patch -p1 < /tflite.patch

ln -snf $(which ${PYTHON}) /usr/local/bin/python

# Run configure.
export TF_NEED_GCP=1
export TF_NEED_HDFS=1
export TF_NEED_S3=1
export TF_NEED_CUDA=1
export TF_NEED_ROCM=0
export TF_CUDA_VERSION=11.3
export TF_CUDNN_VERSION=8.2.1
export TF_CUDA_PATHS=/usr/local/cuda,/usr
export TF_CUDA_COMPUTE_CAPABILITIES=3.5,3.7,5.2,7.0,8.0
export TF_CUDA_CLANG=0
export TF_SET_ANDROID_WORKSPACE=0
export GCC_HOST_COMPILER_PATH=/usr/bin/gcc
export USE_DEFAULT_PYTHON_LIB_PATH=1
# TensorRT build failing as of 2019-12-18, see
# https://github.com/tensorflow/tensorflow/issues/35115
export TF_NEED_TENSORRT=0
export PYTHON_BIN_PATH=$(which python3)
export TMP=/tmp
/usr/local/bin/python configure.py

# Build the pip package and install it
bazel build --config=cuda --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" --config=opt --config=v2 --config=noaws tensorflow/tools/pip_package:build_pip_package
./bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/pip_pkg
ls -al /tmp/pip_pkg
pip --no-cache-dir install --upgrade /tmp/pip_pkg/*.whl
