# Fast Probabilistic Models

# Introduction

This repository introduces Axolotl, an Arm-developed framework that can generate probabilistic models from, virtually, any ML model.

Axolotl provides use cases for:
- Building a model
- Adding uncertainty to a model by inserting Dropout on user-specified layers
  - Dropout used during both training and inference
- Train a model
- Run inference on a model
- Benchmark a model
- Quantize a model

Probabilistic models use sampling to provide uncertainty. Axolotl also provides a tool to generate a new model with a user-specified number of internal branches, this way, we enable generating the desired number of samples with a single inference run, thus decreasing the total inference time by caching the output of layers when possible.

# Why use this repository

Axolotl a framework to enable probabilistic ML experiments. We have included an example with ResNet-18 and CIFAR-10. The main addition is the inclusion of two different functionalities:

1. Automatic addition of Dropout layers into a model
2. Automatic branching of a model with Dropout layers inserted with the step above

Those two functionalites are model-independent, so any other model that is not included in this repository could be used to run probalistic ML experiments with a little effort.

# Repository structure

This repository is structured into model- and dataset-dependant code and independent code.

Model-dependant code include all the needed functionality to create, train, run inference, benchmark and quantize an specific model.

Dataset-dependant code include all the needed functionality to download, prepare and provision a specific dataset (or mockup/random data for testing).

Independent code includes code that can be used without regard of the model or dataset used in your experiments. In this case, the script to add Dropout layers to a given model or the the script to branch a model with Dropout layers are independent of the dataset or model used.

You will also find a folder containing all the necessary scripts to generate a Docker environment that is able to run all the experiments provided in this repository.

Below an example of the code structure. Please note that, in order to improve readiness, not every file in this repository is represented.

```
root_folder
    |- ResNet18.py              # entry point for ResNet-18 experiments
    |- models                   # folder containing model-dependent code
    |   |- resnet18             # folder containing resnet18-dependent code
    |       |- model.py         # contains the necessary code to build and compile a ResNet-18 model
    |       |- trainer.py       # contains the necessary code to train a ResNet-18 model
    |       |- inferencer.py    # contains the necessary code to run inference on a ResNet-18 model
    |       |- benchmarker.py   # contains the necessary code to benchmark a ResNet-18 model
    |       |- converter.py     # contains the necessary code to convert (and quantize) a ResNet-18 model to TFLite
    |- datasets                 # folder containing dataset-specific code
    |   |- cifar10.py           # contains the necessary code to prepare and process CIFAR-10 dataset
    |   |- mockup.py            # contains the necessary code to generate random data valid for a given model
    |- common                   # folder containing model-independent code
        |- mcdo.py              # contains the necessary code to add MCDO to any model
        |- brancher.py          # contains the necessary code to branch any model
        |- utils.py             # contains different useful code for different stuff
```

# Installation

Refer to [INSTALL.md](INSTALL.md).

# Using the framework

Refer to [RUNNING.md](RUNNING.md).

# Pre-trained models

We provide a set of pre-trained ResNet-18 models with and without MCDO. These models are located inside the [experiment_models](experiment_models) folder. The folder contains the following models:

- TensorFlow models
  - FP32 vanilla ResNet-18
  - FP32 Last-Layer MCDO ResNet-18
  - FP32 Partial MCDO ResNet-18
  - FP32 Full MCDO ResNet-18
- TensorFlow Lite models
  - FP32 and INT8 vanilla ResNet-18
  - FP32 and INT8 Last-Layer MCDO ResNet-18
  - FP32 and INT8 Partial MCDO ResNet-18
  - FP32 and INT8 Full MCDO ResNet-18

For a full guide on how those models were generated and the results one should expect from them, please refer to [EXAMPLE.md](EXAMPLE.md).

# License

This project is licensed under Apache-2.0.

This project includes some third-party code under other open source licenses. For more information, see `LICENSE`.

# Contributions / Pull Requests

Contributions are accepted under Apache-2.0. Only Submit contributions where you have authored all of the code. If you do this on work time, make sure you have your employer's approval.
