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
import numbers
from tensorflow.python.ops import nn
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.ops import array_ops

@tf.keras.utils.register_keras_serializable()
class InferenceDropout(tf.keras.layers.Layer):
    def __init__(self, rate, noise_shape=None, seed=None, **kwargs):
        super(InferenceDropout, self).__init__(**kwargs)
        self.rate = rate
        self.noise_shape = noise_shape
        self.seed = seed
        self.supports_masking = True

    def _get_noise_shape(self, inputs):
        if self.noise_shape is None:
            return None

        concrete_inputs_shape = array_ops.shape(inputs)
        noise_shape = []
        for i, value in enumerate(self.noise_shape):
            noise_shape.append(concrete_inputs_shape[i] if value is None else value)
        return ops.convert_to_tensor_v2_with_dispatch(noise_shape)

    def call(self, inputs):
        return nn.dropout(inputs,
                          noise_shape=self._get_noise_shape(inputs),
                          seed=self.seed,
                          rate=self.rate)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
                'rate': self.rate,
                'noise_shape': self.noise_shape,
                'seed': self.seed
                }
        base_config = super(InferenceDropout, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

