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

import tensorflow as tf
import numpy as np
import time
import os

class LearningController(tf.keras.callbacks.Callback):
    def __init__(self, num_epoch=0, learn_minute=0, initial_lr=0):
        self.num_epoch = num_epoch
        self.initial_lr = initial_lr
        self.learn_second = learn_minute * 60
        if self.learn_second > 0:
            print("Learning rate is controlled by time")
        elif self.num_epoch > 0:
            print("Learning rate is controlled by epoch")

    def on_train_begin(self, logs=None):
        if self.learn_second > 0:
            self.start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        if self.learn_second > 0:
            current_time = time.time()

            if current_time - self.start_time > self.learn_second:
                self.model.stop_training = True
                print("Time is up")
                return

            if current_time - self.start_time > self.learn_second / 2:
                self.model.optimizer.lr = self.initial_lr * 0.1
            if current_time - self.start_time > self.learn_second * 3 / 4:
                self.model.optimizer.lr = self.initial_lr * 0.01

        elif self.num_epoch > 0:
            print("Epoch         = {}".format(epoch))
            if epoch > 180:
                print("Epoch is greater than 180, scaling by 0.0005")
                self.model.optimizer.lr = self.initial_lr * 0.0005
            elif epoch > 160:
                print("Epoch is greater than 160, scaling by 0.001")
                self.model.optimizer.lr = self.initial_lr * 0.001
            elif epoch > 120:
                print("Epoch is greater than 120, scaling by 0.01")
                self.model.optimizer.lr = self.initial_lr * 0.01
            elif epoch > 80:
                print("Epoch is greater than 80, scaling by 0.1")
                self.model.optimizer.lr = self.initial_lr * 0.1

        print("New LR        = {:.2e}".format(self.model.optimizer.lr.value()))

class Trainer:
    def configure(self, train_inputs, val_inputs, num_train, num_val, ckpt_path, batch_size, epochs, initial_learning_rate, tensorboard_logdir=None):
        # Dataset related
        self.train_inputs = train_inputs
        self.val_inputs = val_inputs
        self.num_train_inputs = num_train
        self.num_val_inputs = num_val

        # Batch size
        self.batch_size = batch_size

        # Epoch and step related
        self.epochs = epochs

        # Initial learning rate to use
        self.initial_lr = initial_learning_rate

        # Checkpoint related
        self.ckpt_path = ckpt_path
        if not os.path.exists(os.path.dirname(self.ckpt_path)):
            os.makedirs(os.path.dirname(self.ckpt_path))
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=self.ckpt_path,
                                                         save_weights_only=True,
                                                         monitor='val_loss',
                                                         mode='min',
                                                         save_best_only=True,
                                                         save_freq='epoch',
                                                         verbose=1)

        lr_callback = LearningController(self.epochs, initial_lr=self.initial_lr)

        if tensorboard_logdir is not None:
            tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_logdir,
                                                                  histogram_freq=1,
                                                                  write_graph=True,
                                                                  write_images=True,
                                                                  update_freq='epoch')

            self.callbacks = [cp_callback, lr_callback, tensorboard_callback]
        else:
            self.callbacks = [cp_callback, lr_callback]

    def train(self, model):
        hist = model.fit(
                  self.train_inputs,
                  epochs=self.epochs,
                  steps_per_epoch = self.num_train_inputs // self.batch_size,
                  validation_data = self.val_inputs,
                  validation_steps = self.num_val_inputs // self.batch_size,
                  callbacks=self.callbacks)

