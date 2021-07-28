The code itself does not require installation apart from its dependencies. Nevertheless, this code makes use of Dropout layers during inference, which is not supported by TensorFlow Lite at the moment. In order to support it, we provide a patch for TensorFlow 2.4.1 which adds the needed operators so Dropout layers can be used during inference while running a TensorFlow Lite model. Since this patch would need a TensorFlow compilation from scratch, we provide a set of Docker environments for convenience.

# Dependencies

For now, only `Python 3.8` or higher is needed. Further dependencies will be stated at their respective sections. If using the Docker environment, you will of course need Docker and, if using Nvidia's GPU inside Docker, the [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker).

# Docker environment

We provide two different Docker environments, one for x86\_64 systems and another for armv8-a platforms. For both cases, CUDA support is provided as well. The Dockerfiles provided are based on the ones distributed with TensorFlow's code.

## Docker image generation

We provide two different scripts per supported architecture. One uses a prebuilt and prepatched Tensorflow 2.4.1, the other builds Tensorflow from scratch after applying the patch.

### Prebuilt Tensorflow

In order to use the prebuilt Tensorflow, you will need to fetch `.whl` files contained in this repository using Git LFS, for this execute the command below. Please note that you will need to [install Git LFS in your system](https://github.com/git-lfs/git-lfs/tree/v2.13.3#installing).

```bash
git lfs pull
```

Once this command finishes, then you can start generating the Docker image:

```bash
cd docker_env

# For Armv8-a systems
./aarch64-buildPreBuiltImage.sh

# For x86_64 systems
./x86_64-buildPreBuiltImage.sh
```

This step should take a bit of time. By the end, you will have a docker image that you can use already.

### Building TensorFlow

We provide convenience script that do the hard work.

```bash
cd docker_env

# For Armv8-a systems
./aarch64-buildBuilderImage.sh

# For x86_64 systems
./x86_64-buildBuilderImage.sh
```

The commands above take a considerable amount of time to complete since Tensorflow will be built from scratch. Rest, take a cup of coffee/tea and come back later.

## Running the Docker image

We provide a convenience script for starting the container.

```bash
./runContainer.sh
```

At this point, you should have a prompt inside the Docker environment using the same `UID` and `GID` as your user account. The docker image mounts the repository folder at `/probabilistic-classification`.

# Virtual Environment

## System dependencies

First of all, we will install some packages using the system repositories. We assume an Ubuntu distribution for the following command. For other distributions, please check the package's equivalents.

```bash
sudo apt update
sudo apt install -y build-essential curl git git-lfs wget openjdk-8-jdk python3.8 python3-dev virtualenv swig
```
## Generate Python virtual environment

Create a Python virtual environment with the command below. Load the environment once created.

```bash
mkdir python_env && cd python_env
python -m venv .
source ./bin/activate
```

## Python dependencies

There are a bunch of Python dependencies that need to be installed.

```bash
python -m pip install --upgrade pip setuptools
python -m pip install Pillow h5py \
                      keras_preprocessing matplotlib \
                      mock 'numpy<1.19.0' \
                      scipy sklearn \
                      pandas future \
                      portpicker seaborn enum34
```

## Build Tensorflow from sources

Since we need to patch Tensorflow to support Dropout layers at inference time on TFLite models, we will build Tensorflow from scratch. The patch provided in this repository targets release 2.4.1 of Tensorflow. We will download that specific version source code and apply the patch.

```bash
git clone https://github.com/tensorflow/tensorflow -b v2.4.1 tensorflow-2.4.1
cd tensorflow-2.4.1 && patch -p1 < docker_env/required_files/tf-v2.4.1_tflite_randomUniform.patch
```

Now we will install Bazel. The easiest way to do this is by downloading the binary file and including it in your `PATH`.

```bash
# For aarch64 systems (you will need root for the following command)
wget -O /usr/local/bin/bazel "https://github.com/bazelbuild/bazel/releases/download/3.5.1/bazel-3.5.1-linux-arm64"

# For x86_64 systems (you will need root for the following command)
wget -O /usr/local/bin/bazel "https://github.com/bazelbuild/bazel/releases/download/3.5.1/bazel-3.5.1-linux-x86_64"

# Give execution permissions to the bazel binary
chmod +x /usr/local/bin/bazel
```

At this point, you can build Tensorflow as normal. Please refer to the [official documentation](https://www.tensorflow.org/install/source#configure_the_build).

After Tensorflow is installed, we only need to install Tensorflow Datasets Python package.

```bash
python -m pip install tensorflow_datasets
```

Now you should have a working environment that can run this repository.
