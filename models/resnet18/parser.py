"""
SPDX-License-Identifier: Apache-2.0

Copyright (C) 2021, Arm Limited and contributors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import argparse

valid_corruptions = [
        'brightness',
        'contrast',
        'defocus_blur',
        'elastic',
        'fog',
        'frost',
        'frosted_glass_blur',
        'gaussian_blur',
        'gaussian_noise',
        'impulse_noise',
        'jpeg_compression',
        'motion_blur',
        'pixelate',
        'saturate',
        'shot_noise',
        'snow',
        'spatter',
        'speckle_noise',
        'zoom_blur'
        ]

def parse_args():
    # Top-level parser
    parser = argparse.ArgumentParser(description='Entry point')
    
    subparsers = parser.add_subparsers(title='subcommands',
                                       description='valid subcommands',
                                       required=True,
                                       dest='command')

    ################################################################################
    # Parser for command build
    ################################################################################
    build_parser = subparsers.add_parser('build', help='Subcommand for building a ResNet18 model')
    build_parser.add_argument('--save_to',
                              type=str,
                              default='./saved_models',
                              help='Path to folder where to store the model. Folder will be created if it does not exist')
    build_parser.add_argument('--save_filename',
                              type=str, default='resnet18',
                              help='Filename to give to the model that will be saved. Saved model will be in <save_to>/<save_filename>{.save_format}')
    build_parser.add_argument('--save_format',
                              type=str,
                              choices=['h5', 'tf'],
                              default='h5',
                              help='Format to use to save the model')
    ################################################################################
    ################################################################################
    ################################################################################

    ################################################################################
    # Parser for command train
    ################################################################################
    train_parser = subparsers.add_parser('train', help='Subcommand for training ResNet18 models')
    train_parser.add_argument('--model',
                              type=str,
                              required=True,
                              help='Path to ResNet18 saved model to load. If not provided a new model will be created')
    train_parser.add_argument('--save_to',
                              type=str,
                              default='./saved_models',
                              help='Path to folder where to store the model. Folder will be created if it does not exist')
    train_parser.add_argument('--save_filename',
                              type=str, default='resnet18',
                              help='Filename to give to the model that will be saved. Saved model will be in <save_to>/<save_filename>{.save_format}')
    train_parser.add_argument('--save_format',
                              type=str,
                              choices=['h5', 'tf'],
                              default='h5',
                              help='Format to use to save the model')
    train_parser.add_argument('--epochs',
                              type=int,
                              default=20,
                              help='Number of epochs to train the model')
    train_parser.add_argument('--batch',
                              type=int,
                              default=128,
                              help='Batch size to use during training')
    train_parser.add_argument('--initial_learning_rate',
                              type=float,
                              default=0.000717,
                              help='Initial learning rate to use')
    train_parser.add_argument('--ckpt_path',
                              type=str,
                              default='./checkpoints/resnet18/cp_best.ckpt',
                              help='Path where to store checkpoints')
    train_parser.add_argument('--tensorboard_logdir',
                              type=str,
                              help='If set to a directory, Tensorboard profiling will be enabled, storing logs in the specified log dir')
    ################################################################################
    ################################################################################
    ################################################################################

    ################################################################################
    # Parser for command mcdo
    ################################################################################
    mcdo_parser = subparsers.add_parser('mcdo', help='Subcommand to add MCDO to the model')
    mcdo_parser.add_argument('--model',
                             type=str,
                             required=True,
                             help='Path to ResNet18 saved model to load. If not provided a new model will be created')
    mcdo_parser.add_argument('--save_to',
                             type=str,
                             default='./saved_models',
                             help='Path to folder where to store the model. Folder will be created if it does not exist')
    mcdo_parser.add_argument('--save_filename',
                             type=str, default='mcdo_resnet18',
                             help='Filename to give to the model that will be saved. Saved model will be in <save_to>/<save_filename>{.save_format}')
    mcdo_parser.add_argument('--save_format',
                             type=str,
                             choices=['h5', 'tf'],
                             default='h5',
                             help='Format to use to save the model')
    mcdo_parser.add_argument('--starting_layer',
                             type=str,
                             required=True,
                             help='Starting layer in which to start adding MCDO')
    mcdo_parser.add_argument('--droprate',
                             type=float,
                             default=0.10,
                             help='Dropout rate to use')
    ################################################################################
    ################################################################################
    ################################################################################

    ################################################################################
    # Parser for command branch
    ################################################################################
    branch_parser = subparsers.add_parser('branch', help='Subcommand to branch the model')
    branch_parser.add_argument('--model',
                             type=str,
                             required=True,
                             help='Path to ResNet18 saved model to load. If not provided a new model will be created')
    branch_parser.add_argument('--save_to',
                             type=str,
                             default='./saved_models',
                             help='Path to folder where to store the model. Folder will be created if it does not exist')
    branch_parser.add_argument('--save_filename',
                             type=str, default='branched_resnet18',
                             help='Filename to give to the model that will be saved. Saved model will be in <save_to>/<save_filename>{.save_format}')
    branch_parser.add_argument('--save_format',
                             type=str,
                             choices=['h5', 'tf'],
                             default='h5',
                             help='Format to use to save the model')
    branch_parser.add_argument('--n_branches',
                               type=int,
                               choices=range(2,11),
                               default=5,
                               help='How many branches to generate')
    ################################################################################
    ################################################################################
    ################################################################################

    ################################################################################
    # Parser for command inference
    ################################################################################
    inference_parser = subparsers.add_parser('inference', help='Subcommand to run inference on the model')
    inference_parser.add_argument('--model',
                                  type=str,
                                  required=True,
                                  help='Path to ResNet18 saved model to load')
    inference_parser.add_argument('--corruption',
                                  type=str, choices=valid_corruptions,
                                  help='Name of the corruption type to use with CIFAR-10')
    inference_parser.add_argument('--corruption_level',
                                  type=int,
                                  choices=[1,2,3,4,5],
                                  default=1,
                                  help='Level of corruption to use')
    inference_parser.add_argument('--batch',
                                  type=int,
                                  default=128,
                                  help='Batch size to use during training')
    inference_parser.add_argument('--no-header',
                                  action='store_true',
                                  help='Do not print header on the output. Useful to generate CSV files from multiple runs')
    ################################################################################
    ################################################################################
    ################################################################################

    ################################################################################
    # Parser for command benchmark
    ################################################################################
    benchmark_parser = subparsers.add_parser('benchmark', help='Subcommand to benchmark the model')
    benchmark_parser.add_argument('--model',
                                  type=str,
                                  required=True,
                                  help='Path to ResNet18 saved model to load')
    benchmark_parser.add_argument('--repeats',
                                  type=int,
                                  default=100,
                                  help='Number of times to run inference')
    benchmark_parser.add_argument('--batch',
                                  type=int,
                                  default=32,
                                  help='Batch size to use during benchmarking')
    benchmark_parser.add_argument('--no-header',
                                  action='store_true',
                                  help='Do not print header on the output. Useful to generate CSV files from multiple runs')
    ################################################################################
    ################################################################################
    ################################################################################

    ################################################################################
    # Parser for command convert
    ################################################################################
    convert_parser = subparsers.add_parser('convert', help='Subcommand to convert the model to TFLite')
    convert_parser.add_argument('--model',
                                type=str,
                                required=True,
                                help='Path to ResNet18 saved model to load')
    convert_parser.add_argument('--save_to',
                                type=str,
                                default='./saved_models',
                                help='Path to folder where to store the model. Folder will be created if it does not exist')
    convert_parser.add_argument('--save_filename',
                                 type=str, default='resnet18',
                                 help='Filename to give to the model that will be saved. Saved model will be in <save_to>/<save_filename>{.save_format}')
    convert_parser.add_argument('--int8',
                                action='store_true',
                                help='Use this flag to use INT8 post-training quantization while converting the model to TFLite')
    ################################################################################
    ################################################################################
    ################################################################################

    return parser.parse_args()
