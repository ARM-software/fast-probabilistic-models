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
from tqdm import tqdm
import numpy as np
import sys

class Inferencer:
    def configure(self, test_inputs, batch_size, model_format):
        self.test_inputs = test_inputs
        self.batch_size = batch_size
        self.samples = -1 # always use the whole dataset
        self.images_in_ds = len(list(self.test_inputs.unbatch().as_numpy_iterator()))
        self.model_format = model_format

        if self.model_format == 'tflite':
            if self.images_in_ds % self.batch_size != 0:
                print("[WARNING] Batch size ({}) should be a divisor of total number of images ({})!".format(self.batch_size, self.images_in_ds))
                while self.images_in_ds % self.batch_size != 0:
                    self.batch_size -= 1
                self.test_inputs = self.test_inputs.unbatch().batch(self.batch_size)
                print("[WARNING] Batch size set to {}".format(self.batch_size))

    def run(self, model):
        true_labels = np.empty(shape=self.images_in_ds, dtype=np.int64)
        if self.model_format == 'tflite':
            # Get I/O details
            input_details = model.get_input_details()
            output_details = model.get_output_details()

            # Resize inputs if batch_size > 1
            if self.batch_size > 1:
                new_shape = [self.batch_size] + list(input_details[0]['shape'][1:])
                model.resize_tensor_input(model.get_input_details()[0]['index'], new_shape)
                input_details = model.get_input_details()
                output_details = model.get_output_details()
            model.allocate_tensors()
            
            # Run inference
            if len(output_details) > 1:
                output_data = np.empty(shape=(len(output_details), self.images_in_ds, 10), dtype=np.float32)
            else:
                output_data = np.empty(shape=(self.images_in_ds, 10), dtype=np.float32)

            all_inputs = self.test_inputs.take(self.samples)
            if self.batch_size > 1:
                i = 0
                for data, label in tqdm(self.test_inputs, total=int(self.images_in_ds/self.batch_size)):
                    for ii, l in enumerate(label):
                        true_labels[i+ii] = l

                    model.set_tensor(input_details[0]['index'], data)
                    model.invoke()

                    if len(output_details) > 1:
                        for j in range(len(output_details)):
                            for ii in range(self.batch_size):
                                output_data[j][i+ii] = model.get_tensor(output_details[j]['index'])[ii]
                    else:
                        for ii in range(self.batch_size):
                            output_data[i+ii] = model.get_tensor(output_details[0]['index'])[ii]

                    i += self.batch_size
            else:
                i = 0
                for data, label in tqdm(all_inputs, total=self.images_in_ds):
                    true_labels[i] = label
                    model.set_tensor(input_details[0]['index'], data)
                    model.invoke()
                    if len(output_details) > 1:
                        for ii in range(len(output_details)):
                            output_data[ii][i] = model.get_tensor(output_details[ii]['index'])[0]
                    else:
                        output_data[i] = model.get_tensor(output_details[0]['index'])[0]
                    i += 1

            return output_data, true_labels
        else:
            output_data = model.predict(self.test_inputs.take(self.samples), verbose=1)
            # Check if we are running a MCDO model with one output per branch
            # If it is, cast it to a numpy array of shape (n_branches, sampes, 10)
            if isinstance(output_data, list):
                output_data = np.array(output_data)
            # Gather true labels
            i = 0
            for data, label in self.test_inputs.unbatch().take(self.samples):
                true_labels[i] = label
                i += 1
            return output_data, true_labels
