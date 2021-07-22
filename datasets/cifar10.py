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
import tensorflow_datasets as tfds
import numpy as np

import sys

class CIFAR10ImageDataGenerator(tf.keras.preprocessing.image.ImageDataGenerator):
    def __init__(self, cutout_mask_size=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cutout_mask_size = cutout_mask_size

    def cutout(self, x, y):
        return np.array(list(map(self._cutout, x))), y

    def _cutout(self, image_origin):
        image = np.copy(image_origin)
        mask_value = image.mean()

        h, w, _ = image.shape
        top    = np.random.randint(0 - self.cutout_mask_size // 2, h - self.cutout_mask_size)
        left   = np.random.randint(0 - self.cutout_mask_size // 2, w - self.cutout_mask_size)
        bottom = top  + self.cutout_mask_size
        right  = left + self.cutout_mask_size
        
        top  = max(0, top)
        left = max(0, left)

        image[top:bottom, left:right, :].fill(mask_value)
        return image

    def flow(self, *args, **kwargs):
        batches = super().flow(*args, **kwargs)

        while True:
            batch_x, batch_y = next(batches)

            if self.cutout_mask_size > 0:
                result = self.cutout(batch_x, batch_y)
                batch_x, batch_y = result

            yield (batch_x, batch_y)


def normalize_img(image, label):
    return tf.cast(image, tf.float32) / 255., label

class CIFAR10:
    def __init__(self):
        self.ds_name = 'cifar10'

    def configure(self, batch_size, corruption_type=None, corruption_level=None):
        self.batch = batch_size
        if corruption_type is not None:
            self.ds_corr = corruption_type
            self.ds_corr_lvl = corruption_level if corruption_level is not None else 1
            self.ds_name = "cifar10_corrupted/{}_{}".format(self.ds_corr, self.ds_corr_lvl)
        else:
            self.ds_corr = None
            self.ds_corr_lvl = None

    def prepare(self):
        if self.ds_corr is None:
            # Load the dataset, the idea is to have:
            #   - 90% of training images for training
            #   - 10% of training images for training validation
            #   - 100% of test images for inference
            if int(tfds.__version__.split('.')[0]) > 1:
                split = [
                        'train[:90%]',
                        'train[90%:]'
                        ]
            else:
                split = [
                        tfds.Split.TRAIN.subsplit(tfds.percent[:90]),
                        tfds.Split.TRAIN.subsplit(tfds.percent[90:])
                        ]
            (ds_train,
             ds_val), self.ds_info = tfds.load(name=self.ds_name,
                                               split=split,
                                               shuffle_files=True,
                                               as_supervised=True,
                                               with_info=True,
                                               batch_size=-1)

            self.ds_test = tfds.load(name=self.ds_name,
                                     split=['test'],
                                     shuffle_files=False,
                                     as_supervised=True,
                                     with_info=False)

            self.train_examples = int(self.ds_info.splits['train'].num_examples * 0.9)
            self.val_examples   = int(self.ds_info.splits['train'].num_examples * 0.1)
            self.test_examples  = int(self.ds_info.splits['test'].num_examples)

            # Up to here, ds_{train,val,test} are a 2-elem list with the example tensors in the first index
            # and the label tensor in the second index
            self.ds_train_x = ds_train[0] / 255
            self.ds_train_y = ds_train[1]
            self.ds_val_x = ds_val[0] / 255
            self.ds_val_y = ds_val[1]

            datagen_parameters = {"horizontal_flip": True,
                                  "width_shift_range": 0.1,
                                  "height_shift_range": 0.1,
                                  "cutout_mask_size": 16}
            self.train_datagen = CIFAR10ImageDataGenerator(**datagen_parameters)
            self.val_datagen = tf.keras.preprocessing.image.ImageDataGenerator()

            self.ds_test = self.ds_test[0].map(normalize_img)
            self.ds_test = self.ds_test.batch(self.batch)
            self.ds_test = self.ds_test.cache()
            self.ds_test = self.ds_test.prefetch(tf.data.experimental.AUTOTUNE)
        else:
            # On CIFAR-10 corrupted, there's only a test split
            (self.ds_test), self.ds_info = tfds.load(name=self.ds_name,
                                                     split=['test'],
                                                     shuffle_files=False,
                                                     as_supervised=True,
                                                     with_info=True)

            self.test_examples = int(self.ds_info.splits['test'].num_examples)

            self.ds_test = self.ds_test[0].map(normalize_img)
            self.ds_test = self.ds_test.batch(self.batch)
            self.ds_test = self.ds_test.cache()
            self.ds_test = self.ds_test.prefetch(tf.data.experimental.AUTOTUNE)

    def get_train_split(self):
        if self.ds_corr is None:
            return self.train_datagen.flow(self.ds_train_x, self.ds_train_y, batch_size=self.batch), self.train_examples
        else:
            # Corrupted CIFAR-10 doesn't have a train split
            return None

    def get_validation_split(self):
        if self.ds_corr is None:
            return self.val_datagen.flow(self.ds_val_x, self.ds_val_y, batch_size=self.batch), self.val_examples
        else:
            # Corrupted CIFAR-10 doesn't have a validation split
            return None

    def get_test_split(self):
        return self.ds_test

    def get_supervised_info(self):
        return self.ds_info

    def get_dataset_name(self):
        return self.name

