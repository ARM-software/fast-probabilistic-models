# Docker environments

The files contained in this folder are meant to create Docker container with all the necessary software stack to run the framework provided by this repository.

The TensorFlow package within the container will support Nvidia GPUs as long as the Nvidia Container Toolkit is installed.

## Building the image

### If pre-built TensorFlow wheel package is not available

```bash
./x86_64-buildBuilderImage.sh
# or
./aarch64-buildBuilderImage.sh
```

### If pre-built TensorFlow wheel package is available

```bash
./x86_64-buildPreBuiltImage.sh
# or
./aarch64-buildPreBuiltImage.sh
```

## Running the container

```bash
./runContainer.sh
```
