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

import numpy as np

class MOCKUP:
    def __init__(self):
        self.ds_name = 'mockup'

    def configure(self, shape, batch=32, dtype=np.float32):
        self.batch = batch
        self.shape = shape
        self.dtype = dtype

    def prepare(self):
        # In TFlite, shape is returned as list, so if there's only one input, shape[0] won't be a list
        if isinstance(self.shape, list) and isinstance(self.shape[0], list):
            self.data = []
            for shape in self.shape:
                new_shape = [self.batch, *shape[1:]]
                self.data.append(np.array(np.random.random_sample(new_shape),
                                          dtype=self.dtype))
        else:
            new_shape = [self.batch, *self.shape[1:]]
            self.data = np.array(np.random.random_sample(new_shape),
                                 dtype=self.dtype)

    def get_train_split(self):
        return None

    def get_validation_split(self):
        return None

    def get_test_split(self):
        return self.data

    def get_supervised_info(self):
        return None

    def get_dataset_name(self):
        return self.name

