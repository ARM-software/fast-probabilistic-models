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

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from datasets import cifar10, mockup
from models.resnet18 import model, trainer, parser, inferencer, benchmarker, converter
from common import mcdo, brancher
from common import utils
import sys
import scipy.special as sc
import numpy as np

def build(args):
    my_model = model.Model()
    # Build the model
    my_model.build()

    # Save the model
    # Check that folder exists, if not create it
    if not os.path.exists(args.save_to):
        os.makedirs(args.save_to)
    my_model.save(filename=args.save_filename,
                  path=args.save_to,
                  save_format=args.save_format)

def train(args):
    my_model = model.Model()
    # Load the model
    my_model.load(args.model)

    if my_model.model_format == 'tflite':
        utils.error_and_exit("Can't train a TF Lite model, please use an H5 or TF model")

    # Prepare inputs
    inputs = cifar10.CIFAR10()
    inputs.configure(args.batch)
    inputs.prepare()

    # Train the model
    my_trainer = trainer.Trainer()
    train_data = inputs.get_train_split()
    val_data = inputs.get_validation_split()
    init_lr = args.initial_learning_rate

    # Now that we know the initial learning rate, compile the model
    my_model.compile(initial_lr=init_lr)

    my_trainer.configure(train_inputs=train_data[0],
                         val_inputs=val_data[0],
                         num_train=train_data[1],
                         num_val=val_data[1],
                         ckpt_path=args.ckpt_path,
                         epochs=args.epochs,
                         batch_size=args.batch,
                         initial_learning_rate=init_lr,
                         tensorboard_logdir=args.tensorboard_logdir)
    my_trainer.train(my_model.model)
    # Load best weights saved during training
    my_model.load_weights(args.ckpt_path)

    # Save the model
    # Check that folder exists, if not create it
    if not os.path.exists(args.save_to):
        os.makedirs(args.save_to)
    my_model.save(filename=args.save_filename,
                  path=args.save_to,
                  save_format=args.save_format)

def add_mcdo(args):
    my_model = model.Model()
    my_model.load(args.model)

    if my_model.model_format == 'tflite':
        utils.error_and_exit("Can't add MCDO to a TF Lite model, please use an H5 or TF model")
    
    # Configure MCDO
    mdo = mcdo.MCDO()
    mdo.configure(args.starting_layer,
                  args.droprate)

    # Add MCDO to the model
    my_model.model = mdo.add_mcdo(my_model.model)

    # Save the model
    # Check that folder exists, if not create it
    if not os.path.exists(args.save_to):
        os.makedirs(args.save_to)
    my_model.save(filename=args.save_filename,
                  path=args.save_to,
                  save_format=args.save_format)

def branch(args):
    my_model = model.Model()
    my_model.load(args.model)

    if my_model.model_format == 'tflite':
        utils.error_and_exit("Can't branch a TF Lite model, please use an H5 or TF model")

    # Get the initial learning rate of the model for later
    init_lr = my_model.get_lr()

    # Configure branch
    b = brancher.Branch()
    b.configure(args.n_branches)

    # Branch the model
    my_model.model = b.branch(my_model.model)

    # The branched model is in fact a new model, we need to re-compile it
    my_model.compile(init_lr)

    # Save the mode
    # Check that folder exists, if not create it
    if not os.path.exists(args.save_to):
        os.makedirs(args.save_to)
    my_model.save(filename=args.save_filename,
                  path=args.save_to,
                  save_format=args.save_format)

def inference(args):
    my_model = model.Model()
    my_model.load(args.model)

    # Prepare inputs
    inputs = cifar10.CIFAR10()
    inputs.configure(args.batch,
                     corruption_type=args.corruption,
                     corruption_level=args.corruption_level)
    inputs.prepare()
    
    inference = inferencer.Inferencer()
    inference.configure(inputs.get_test_split(), args.batch, my_model.model_format)
    predictions, true_labels = inference.run(my_model.model)

    # Process predictions
    # ResNet-18 outputs a (N, 1, 10) where N == samples
    # If using MCDO, then it will output (M, N, 1, 10), where M == number of branches and N == samples
    if predictions.ndim == 3:
        preds, lower_lim, upper_lim, std_preds = utils.calculate_output(predictions)
    elif predictions.ndim == 2:
        preds = np.asarray([sc.softmax(output_sample, axis=0) for output_sample in predictions])

    metrics = utils.metrics_from_stats({'labels': true_labels, 'probs': preds})

    corruption_type = "no_corruption" if args.corruption is None else args.corruption
    corruption_lvl  = 0 if args.corruption is None else args.corruption_level
    if not args.no_header:
        print("corruption,corruption_level,accuracy,brier_score,log_prob,ece")
    print("{},{},{},{},{},{}".format(corruption_type,
                                     corruption_lvl,
                                     metrics['accuracy'],
                                     metrics['brier_score'],
                                     metrics['log_prob'],
                                     metrics['ece']))

def benchmark(args):
    my_model = model.Model()
    my_model.load(args.model)

    # Prepare inputs
    inputs = mockup.MOCKUP()
    if my_model.model_format == 'tflite':
        model_inputs = my_model.model.get_input_details()
        if len(model_inputs) > 1:
            input_shape = []
            for inp in model_inputs:
                input_shape.append(inp['shape'])
        else:
            input_shape = model_inputs[0]['shape']
    else:
        model_inputs = my_model.model.input
        # We need to consider the possibilty the model more than 1 input
        # If so, send a list of shapes instead of only one
        if isinstance(model_inputs, list):
            input_shape = []
            for i in model_inputs:
                input_shape.append(i.get_shape)
        else:
            input_shape = model_inputs.get_shape()
    inputs.configure(batch=args.batch,
                     shape=input_shape)
    inputs.prepare()

    # Prepare benchmark
    benchmark = benchmarker.Benchmarker()
    benchmark.configure(inputs=inputs.get_test_split(),
                        batch=args.batch,
                        repeats=args.repeats,
                        model_format=my_model.model_format,
                        no_header=args.no_header)

    # Run benchmark
    benchmark.run(my_model.model, my_model.model_size)

def convert(args):
    my_model = model.Model()
    my_model.load(args.model)

    if my_model.model_format == 'tflite':
        utils.error_and_exit("Can't convert a TF Lite model, please use an H5 or TF model")

    q = converter.Converter()
    q.configure(to_int8=args.int8)
    tflite_model = model.Model()
    tflite_model.model = q.convert(my_model.model)

    # Save the model
    # Check that folder exists, if not create it
    if not os.path.exists(args.save_to):
        os.makedirs(args.save_to)
    tflite_model.save(filename=args.save_filename,
                      path=args.save_to,
                      save_format='tflite')

if __name__ == "__main__":
    args = parser.parse_args()

    my_model = model.Model()
    if args.command == 'build':
        build(args)

    elif args.command == 'train':
        train(args)

    elif args.command == 'mcdo':
        add_mcdo(args)

    elif args.command == 'branch':
        branch(args)

    elif args.command == 'inference':
        inference(args)

    elif args.command == 'benchmark':
        benchmark(args)

    elif args.command == 'convert':
        convert(args)

    else:
        utils.error_and_exit("You should have not reached here")

