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
import os
from common import utils

def residual_block(inputs, filters):
    x = tf.keras.layers.Conv2D(filters,
                               (3,3),
                               padding='same',
                               kernel_initializer='he_normal',
                               kernel_regularizer=tf.keras.regularizers.l2(1e-4))(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Conv2D(filters,
                               (3,3),
                               padding='same',
                               kernel_initializer='he_normal',
                               kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    #x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Add()([inputs, x])
    return tf.keras.layers.ReLU()(x)

def transitional_block(inputs, filters):
    x = tf.keras.layers.Conv2D(filters,
                               (3,3),
                               padding='same',
                               strides=(2,2),
                               kernel_initializer='he_normal',
                               kernel_regularizer=tf.keras.regularizers.l2(1e-4))(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Conv2D(filters,
                               (3,3),
                               padding='same',
                               kernel_initializer='he_normal',
                               kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = tf.keras.layers.BatchNormalization()(x)

    y = tf.keras.layers.Conv2D(filters,
                               (1,1),
                               padding='same',
                               strides=(2,2),
                               kernel_initializer='he_normal',
                               kernel_regularizer=tf.keras.regularizers.l2(1e-4))(inputs)

    x = tf.keras.layers.Add()([x, y])
    return tf.keras.layers.ReLU()(x)

class Model:
    def __init__(self):
        self.name = 'resnet18'
        self.num_classes = 10

    def build(self):
        filters = 16
        inputs = tf.keras.layers.Input(shape=(32,32,3,), dtype=tf.float32)
        x = tf.keras.layers.Conv2D(filters,
                                   (3,3),
                                   padding='same',
                                   kernel_initializer='he_normal',
                                   kernel_regularizer=tf.keras.regularizers.l2(1e-4))(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)

        for i in range(3):
            x = residual_block(x, filters)
        
        for _ in range(2):
            filters *= 2
            x = transitional_block(x, filters)

            for i in range(2):
                x = residual_block(x, filters)

        x = tf.keras.layers.AveragePooling2D(pool_size=(8,8),
                                             strides=(8,8),
                                             padding='valid')(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(self.num_classes,
                                  kernel_initializer='he_normal')(x)

        self.model = tf.keras.models.Model(inputs=[inputs], outputs=x, name=self.name)

    def get_lr(self):
        return self.model.optimizer.lr.value()

    def compile(self, initial_lr):
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=initial_lr)
        self.loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.metrics = ['sparse_categorical_accuracy']
        self.model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)

    def load(self, model_path):
        self.model_size = os.stat(model_path).st_size / float(2**20)
        head, tail = os.path.splitext(model_path)
        if not tail:
            self.model_format = 'tf'
        elif tail == '.h5':
            self.model_format = 'h5'
        elif tail == '.tflite':
            self.model_format = 'tflite'
        else:
            utils.error_and_exit("Model format not recognized")

        if self.model_format == 'tflite':
            self.model = tf.lite.Interpreter(model_path, num_threads=os.cpu_count())
        else:
            self.model = tf.keras.models.load_model(model_path)
    
    def load_weights(self, ckpt_path):
        latest_ckpt = tf.train.latest_checkpoint(os.path.dirname(ckpt_path))
        self.model.load_weights(latest_ckpt)
    
    def save(self, filename=None, path=None, save_format='h5'):
        if filename is None:
            name = model.name
        if save_format == 'h5':
            filename = filename + '.h5'
        elif save_format == 'tflite':
            filename = filename + '.tflite'
        if path is not None:
            save_path = "{}/{}".format(path, filename)
        else:
            save_path="./{}".format(filename)

        if save_format == 'tflite':
            with open(save_path, 'wb') as f:
                f.write(self.model)
        else:
            self.model.save(save_path, save_format=save_format)
    
