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
import scipy
import timeit
import time
import statistics as st
import os

class Benchmarker:
    def configure(self, inputs, batch, repeats, model_format, no_header):
        self.print_header = False if no_header else True
        self.inputs = inputs
        self.batch = batch
        self.model_format = model_format
        self.repeats = repeats
        if self.model_format == 'tflite':
            self.code = """\
i = 0
while i < len(self.inputs):
    input_data = self.inputs[i:i+self.batch]
    model.set_tensor(input_details[0]['index'], input_data)
    model.invoke()
    i += self.batch
"""
        else:
            self.code = """\
model.predict(self.inputs)
"""

    def run(self, model, model_size):
        # Need to resize inputs if batch > 1 and using TFLite
        if self.model_format == 'tflite':
            input_details = model.get_input_details()
            if self.batch > 1:
                new_shape = [self.batch, *input_details[0]['shape'][1:]]
                model.resize_tensor_input(model.get_input_details()[0]['index'], new_shape)
            model.allocate_tensors()

            # Get input details
            input_details = model.get_input_details()

            total_time = timeit.repeat(stmt=self.code,
                                       number=1,
                                       repeat=self.repeats,
                                       globals = {
                                                  'tf': globals().get('tf'),
                                                  'np': globals().get('np'),
                                                  'self': locals().get('self'),
                                                  'model': locals().get('model'),
                                                  'input_details': locals().get('input_details')
                                       })

        else:
            total_time = timeit.repeat(stmt=self.code,
                                       number=1,
                                       repeat=self.repeats,
                                       globals = {
                                                  'tf': globals().get('tf'),
                                                  'np': globals().get('np'),
                                                  'self': locals().get('self'),
                                                  'model': locals().get('model')
                                       })
        if self.repeats == 1:
            if self.print_header:
                print("model_size[MB],exec_time[sec]")
            print("{:.4f},{:.4f}".format(model_size, total_time))
        else:
            if self.print_header:
                print("model_size[MB],mean[sec],median[sec],max[sec],min[sec],stdev")
            print("{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f}".format(
                model_size,
                st.mean(total_time),
                st.median(total_time),
                max(total_time),
                min(total_time),
                st.stdev(total_time)))

