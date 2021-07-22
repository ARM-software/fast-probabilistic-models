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
import argparse
import tensorflow as tf
import numpy
from common import utils
from common import custom_layers

class Branch:
    def configure(self, n_branches=5):
        self.n_branches = n_branches

    def transfer_weights(self, orig, new):
        for idx, new_layer in enumerate(new.layers):
            new_layer_type = new_layer.__class__.__bases__[0]
            model_type = tf.keras.models.Model
            if new_layer_type == model_type:
                for jdx, l in enumerate(new_layer.layers):
                    new_layer_name = l.get_config()['name'].split('_branch')[0]
                    for ol in orig.layers:
                        orig_layer_name = ol.get_config()['name']
                        orig_layer_type = ol.__class__.__bases__[0]
                        if orig_layer_type == model_type:
                            found = False
                            for ol2 in ol.layers:
                                orig_layer_name = ol2.get_config()['name']
                                if orig_layer_name == new_layer_name:
                                    found = True
                                    new.layers[idx].layers[jdx].set_weights(ol2.get_weights())
                                    break
                                if found:
                                    break
                        else:
                            if orig_layer_name == new_layer_name:
                                new.layers[idx].layers[jdx].set_weights(ol.get_weights())
                                break
            else:
                new_layer_name = new_layer.get_config()['name']
                for ol in orig.layers:
                    orig_layer_name = ol.get_config()['name']
                    orig_layer_type = ol.__class__.__bases__[0]
                    if orig_layer_type == model_type:
                        found = False
                        for ol2 in ol.layers:
                            orig_layer_name = ol2.get_config()['name']
                            if orig_layer_name == new_layer_name:
                                found = True
                                new.layers[idx].set_weights(ol.get_weights())
                                break
                    if orig_layer_name == new_layer_name:
                        new.layers[idx].set_weights(ol.get_weights())
                        break


    def branch(self, model):
        branched_model = []
        for n_layer, layer in enumerate(model.layers):
            # If the layer is a model, then we found the part of the model that we need to branch
            layer_config = layer.get_config()
            layer_type = layer.__class__.__bases__[0]
            if layer_type == tf.keras.models.Model:
                # First of all, create now the backbone model
                # First, check how many outputs the backbone should have
                output_layers = model.get_config()['layers'][n_layer]['inbound_nodes'][0]
                output_layer_names = [ x[0] for x in output_layers ]
                outputs = []
                # Find the indices of those layers in the initial model
                for oname in output_layer_names:
                    for i, layer in enumerate(model.layers):
                        lname = layer.name
                        if lname == oname:
                            outputs.append(model.layers[i].output)

                backbone_model = tf.keras.models.Model(inputs=model.inputs, outputs=outputs, name="backbone")
                for i in range(self.n_branches):
                    config = layer.get_config()
                    config['name'] = "{}_branch_{}".format(config['name'], i)
                    for idx, input_layer in enumerate(config['input_layers']):
                        config['input_layers'][idx][0] = "{}_branch_{}".format(input_layer[0], i)
                    config['output_layers'][0][0] = "{}_branch_{}".format(config['output_layers'][0][0], i)
                    for idx, _layer in enumerate(config['layers']):
                        connections = _layer['inbound_nodes']
                        if connections:
                            connections = connections[0]
                            for jdx, con in enumerate(connections):
                                con_name = config['layers'][idx]['inbound_nodes'][0][jdx][0]
                                config['layers'][idx]['inbound_nodes'][0][jdx][0] = "{}_branch_{}".format(con_name, i)
                        config['layers'][idx]['name'] = "{}_branch_{}".format(config['layers'][idx]['name'], i)
                        config['layers'][idx]['config']['name'] = config['layers'][idx]['name']

                    branched_model.append(tf.keras.models.Model().from_config(config))

        y = backbone_model.output
        x = []
        for i in range(self.n_branches):
            x.append(branched_model[i](y))

        new_model = tf.keras.models.Model(inputs=backbone_model.input, outputs=x, name="mcdo_model")
        
        # At this point, the only thing left to do is copy weights from the original model to the new one
        self.transfer_weights(model, new_model)
        return new_model
    
