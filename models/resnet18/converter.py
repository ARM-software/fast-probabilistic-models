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
from datasets import cifar10

def representative_dataset_gen():
    inputs = cifar10.CIFAR10()
    inputs.configure(batch_size=1)
    inputs.prepare()

    for data in inputs.get_test_split().take(100):
        yield [data[0]]

class Converter:
    def configure(self, to_int8=True):
        self.to_int8=to_int8

    def convert(self, model):
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset_gen
        if self.to_int8:
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
                                                   tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        else:
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
            converter.target_spec.supported_types = [tf.float32]
        converter.inference_input_type = tf.float32
        converter.inference_output_type = tf.float32
        converter.allow_custom_ops = True
        return converter.convert()
