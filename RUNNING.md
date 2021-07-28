# Using this framework

`ResNet18.py` provides different subcommands for different tasks. The workflow for the example provided in this repository would be:

1. Build the model
1. **(Optional)** Add MCDO to the model
1. Train the model
1. **(Optional)** Branch the MCDO model
1. **(Optional)** Convert to TensorFlow Lite model (optionally applying INT8 post-training quantization)
1. Run inference on the model
1. Benchmark the model

The workflow stated above represents a full use of this framework. For example, if you already have a trained ResNet-18 model and you only want to understand how adding MCDO to the model affects accuracy and ECE, you would only do step `2`, `4`, `5` and `6`.

# Fast start

We provide end-to-end scripts for running a full example with ResNet-18 using CIFAR-10 and CIFAR-10-corrupted in [scripts/prepare_resnet18.sh](scripts/prepare_resnet18.sh) and [scripts/inference_resnet18.sh](scripts/inference_resnet18.sh). You can use this scripts as following:

```bash
# Prepare the ResNet18 models by building, adding MCDO, training, branching and conversion to TFLite
./scripts/prepare_resnet18.sh

# Run inference on the TFLite models
./scripts/inference_resnet18.sh tflite
```

After everything is executed, you should have four PDF files with names `fp32_acc.pdf`, `fp32_ece.pdf`, `int8_acc.pdf` and `int8_ece.pdf`.

## Build the model

Building the model is the first step of the experiment. The following commands will create a ResNet-18 model prepared to use with a CIFAR-10 dataset.

### Usage

```bash
usage: ResNet18.py build [-h] [--save_to SAVE_TO] [--save_filename SAVE_FILENAME] [--save_format {h5,tf}]

optional arguments:
  -h, --help            show this help message and exit
  --save_to SAVE_TO     Path to folder where to store the model. Folder will be created if it does not exist
  --save_filename SAVE_FILENAME
                        Filename to give to the model that will be saved. Saved model will be in <save_to>/<save_filename>{.save_format}
  --save_format {h5,tf}
                        Format to use to save the model
```

### Example

```bash
python ResNet18.py build --save_to ./experiment_models --save_filename fp32_vanilla --save_format h5
```

## Add MCDO to the model

The next, and optional, step is to add MCDO to the model. For this, you will need to know first from which branch you want to start adding Dropout layers. You will need the layer name of it. You can get this information by inspecting the model with [Netron](https://netron.app) or by using `tf.keras.models.Model.summary()`.

The script will add Dropout layers to any convolutional or dense layer starting from the layer specified. On top of the that, this script will inform about the overhead added by the dropout layers. An example of this can be found in the [end-to-end example](EXAMPLE.md).

### Usage

```bash
usage: ResNet18.py mcdo [-h] --model MODEL [--save_to SAVE_TO] [--save_filename SAVE_FILENAME] [--save_format {h5,tf}] --starting_layer
                        STARTING_LAYER [--droprate DROPRATE]

optional arguments:
  -h, --help            show this help message and exit
  --model MODEL         Path to ResNet18 saved model to load. If not provided a new model will be created
  --save_to SAVE_TO     Path to folder where to store the model. Folder will be created if it does not exist
  --save_filename SAVE_FILENAME
                        Filename to give to the model that will be saved. Saved model will be in <save_to>/<save_filename>{.save_format}
  --save_format {h5,tf}
                        Format to use to save the model
  --starting_layer STARTING_LAYER
                        Starting layer in which to start adding MCDO
  --droprate DROPRATE   Dropout rate to use
```

### Example

```bash
python ResNet18.py mcdo --model ./experiment_models/fp32_vanilla.h5 --save_to ./experiment_models --save_filename fp32_ll_mcdo --save_format h5 --starting_layer dense --droprate 0.319811
```

## Train the model

After your model is ready (i.e., the model is build and, optionally, MCDO has been added). You can train the model. The script will use the a 10% of the training split of CIFAR-10 as validation data.

### Usage

```bash
usage: ResNet18.py train [-h] --model MODEL [--save_to SAVE_TO] [--save_filename SAVE_FILENAME] [--save_format {h5,tf}] [--epochs EPOCHS]
                         [--batch BATCH] [--initial_learning_rate INITIAL_LEARNING_RATE] [--ckpt_path CKPT_PATH]
                         [--tensorboard_logdir TENSORBOARD_LOGDIR]

optional arguments:
  -h, --help            show this help message and exit
  --model MODEL         Path to ResNet18 saved model to load. If not provided a new model will be created
  --save_to SAVE_TO     Path to folder where to store the model. Folder will be created if it does not exist
  --save_filename SAVE_FILENAME
                        Filename to give to the model that will be saved. Saved model will be in <save_to>/<save_filename>{.save_format}
  --save_format {h5,tf}
                        Format to use to save the model
  --epochs EPOCHS       Number of epochs to train the model
  --batch BATCH         Batch size to use during training
  --initial_learning_rate INITIAL_LEARNING_RATE
                        Initial learning rate to use
  --ckpt_path CKPT_PATH
                        Path where to store checkpoints
  --tensorboard_logdir TENSORBOARD_LOGDIR
                        If set to a directory, Tensorboard profiling will be enabled, storing logs in the specified log dir
```

### Example

```bash
python ResNet18.py train --model ./experiment_models/fp32_ll_mcdo.h5 --save_to ./experiment_models --save_filename fp32_ll_mcdo --save_format h5 --epochs 200 --batch 16 --initial_learning_rate 0.000313
```

## Branch the model

For MCDO models, the idea during inference is to run the model several times and average the outputs. To simplify this, we offer a script to branch the MCDO model. This will create a single model with N MCDO branches (i.e., the model will be composed by a backbone model including the non-MCDO part of it, and N equal branches including the layers of the MCDO part of the model).

After using this script, your model will have as many outputs as branches.

### Usage

```bash
usage: ResNet18.py branch [-h] --model MODEL [--save_to SAVE_TO] [--save_filename SAVE_FILENAME] [--save_format {h5,tf}]
                          [--n_branches {2,3,4,5,6,7,8,9,10}]

optional arguments:
  -h, --help            show this help message and exit
  --model MODEL         Path to ResNet18 saved model to load. If not provided a new model will be created
  --save_to SAVE_TO     Path to folder where to store the model. Folder will be created if it does not exist
  --save_filename SAVE_FILENAME
                        Filename to give to the model that will be saved. Saved model will be in <save_to>/<save_filename>{.save_format}
  --save_format {h5,tf}
                        Format to use to save the model
  --n_branches {2,3,4,5,6,7,8,9,10}
                        How many branches to generate
```

### Example

```bash
python ResNet18.py --model ./experiment_models/fp32_ll_mcdo.h5 --save_to ./experiment_models --save_filename fp32_ll_mcdo --save_format h5 --n_branches 5
```

## Convert to TensorFlow Lite

At this point, if you need it, you can convert the trained model to TensorFlow Lite and, optionally, quantize the model using TFLite post-training quantization.

### Usage

```bash
usage: ResNet18.py convert [-h] --model MODEL [--save_to SAVE_TO] [--save_filename SAVE_FILENAME] [--int8]

optional arguments:
  -h, --help            show this help message and exit
  --model MODEL         Path to ResNet18 saved model to load
  --save_to SAVE_TO     Path to folder where to store the model. Folder will be created if it does not exist
  --save_filename SAVE_FILENAME
                        Filename to give to the model that will be saved. Saved model will be in <save_to>/<save_filename>{.save_format}
  --int8                Use this flag to use INT8 post-training quantization while converting the model to TFLite
```

### Example

```bash
# To generate FP32 TFLite model
python ResNet18.py convert --model ./experiment_models/fp32_ll_mcdo.h5 --save_to ./experiment_models --save_filename fp32_ll_mcdo

# To generate quantized INT8 TFLite model
python ResNet18.py convert --model ./experiment_models/fp32_ll_mcdo.h5 --save_to ./experiment_models --save_filename fp32_ll_mcdo --int8
```

## Run inference on the model

To run inference on the model and get accuracy metrics, along with ECE, Brier scores, etc. Run the `inference` subcommand. For inference, we use the test split of CIFAR-10 and, if specified, the test split of the specific corruption type and level CIFAR-10-C dataset.

### Usage

```bash
usage: ResNet18.py inference [-h] --model MODEL
                             [--corruption {brightness,contrast,defocus_blur,elastic,fog,frost,frosted_glass_blur,gaussian_blur,gaussian_noise,impulse_noise,jpeg_compression,motion_blur,pixelate,saturate,shot_noise,snow,spatter,speckle_noise,zoom_blur}]
                             [--corruption_level {1,2,3,4,5}] [--batch BATCH] [--no-header]

optional arguments:
  -h, --help            show this help message and exit
  --model MODEL         Path to ResNet18 saved model to load
  --corruption {brightness,contrast,defocus_blur,elastic,fog,frost,frosted_glass_blur,gaussian_blur,gaussian_noise,impulse_noise,jpeg_compression,motion_blur,pixelate,saturate,shot_noise,snow,spatter,speckle_noise,zoom_blur}
                        Name of the corruption type to use with CIFAR-10
  --corruption_level {1,2,3,4,5}
                        Level of corruption to use
  --batch BATCH         Batch size to use during training
  --no-header           Do not print header on the output. Useful to generate CSV files from multiple runs
```

### Example

```bash
# Use CIFAR-10
python ResNet18.py inference --model ./experiment_models/int8_ll_mcdo.tflite --batch 1

# Use CIFAR-10-C, with corruption saturate and corruption level 3
python ResNet18.py inference --model ./experiment_models/int8_ll_mcdo.tflite --batch 1 --corruption saturate --corruption_level 3
```

### Plot inference results

We provide a small script to plot accuracy and ECE metrics obtained the output of the inference subcommand as a CSV file. You can use this script as:

```bash
python plot.py --input {csv_file_1} {csv_file_2} ... {csv_file_N}
```

For ease of programming, we added a few restrictions to it:

- All CSV files must contain the same number of lines
- Each CSV file must contain each entry ordered
  - i.e., for each corruption type, all the severity levels must be ordered from lowest to highest and contiguous
- CSV files should be named `${prec}_${model_name}.csv` (e.g., fp32_vanilla.csv, fp32_ll_mcdo.csv)
- Each CSV file has a header in the first line
- The content of each CSV file is `corruption,corruption_level,accuracy,brier_score,log_prob,ece`
  - No corruption must be identified with `no_corruption` in the corruption column
  - Corruption level when no corruption is used must be identified as `0`
- All CSV files need to have the same precision (e.g., fp32, int8)
- Name of the generated PDF's will be `${prec}_acc.pdf` for the accuracy plot and `${prec}_ece.pdf` for the ECE plot

## Benchmark the model

We offer a small tool to benchmark the model with synthetic data. This script will generate random data of a valid shape for each input of the output model and will time inference runs. No accuracy metrics are provided, only inference time.

### Usage

```bash
usage: ResNet18.py benchmark [-h] --model MODEL [--repeats REPEATS] [--batch BATCH] [--no-header]

optional arguments:
  -h, --help         show this help message and exit
  --model MODEL      Path to ResNet18 saved model to load
  --repeats REPEATS  Number of times to run inference
  --batch BATCH      Batch size to use during benchmarking
  --no-header        Do not print header on the output. Useful to generate CSV files from multiple runs
```

### Example

```bash
python ResNet18.py benchmark --model ./experiment_models/int8_ll_mcdo.tflite --repeats 1000 --batch 32
```
