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
from tensorflow.python.framework.convert_to_constants import (
        convert_variables_to_constants_v2_as_graph
        )
import numpy
from common import utils
from common import custom_layers

import sys

def get_flops(model):
    batch_size = 1

    inputs = [tf.TensorSpec([batch_size] + inp.shape[1:], inp.dtype) for inp in model.inputs]
    real_model = tf.function(model).get_concrete_function(inputs)
    frozen_func, _ = convert_variables_to_constants_v2_as_graph(real_model)

    run_metadata = tf.compat.v1.RunMetadata()
    opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
    opts['output'] = "none"
    flops = tf.compat.v1.profiler.profile(graph=frozen_func.graph,
                                          run_meta=run_metadata,
                                          cmd="scope",
                                          options=opts)
    return flops.total_float_ops

class MCDO:
    def configure(self, start_layer, droprate):
        self.start_layer = start_layer
        if droprate < 0.0 and droprate >= 1.0:
            utils.error_and_exit("Dropout rate must be in the interval [0.0, 1.0)")
        self.droprate = droprate

    def add_mcdo(self, model):
        layersIdx = []
        first_layer_for_dropout_found = False
        for i in range(len(model.layers)):
            layer = model.layers[i]
            isConvOrDense = True if type(layer) in [tf.keras.layers.Conv2D, tf.keras.layers.Dense] else False
            if first_layer_for_dropout_found:
                if isConvOrDense:
                    layersIdx.append(i)
            else:
                if layer.name == self.start_layer:
                    first_layer_for_dropout_found = True
                    if isConvOrDense:
                        layersIdx.append(i)

        self.inserted_dropout_layers = len(layersIdx)
        
        # Generate the backbone model (i.e., a model with layers from 0 to layer with the first dropout
        backboneModel = tf.keras.models.Model(inputs=model.input, outputs=model.layers[layersIdx[0]-1].output, name='backbone_model')
        
        # Copy the rest of the layers into new layers
        # We need to change the name of the layer so it won't collide with the original model
        # We only generate 1 branch for now, at the end, we will replicate that branch to generate the other 4 branches
        layers = model.layers
        split_layers = []
        split_layer_names = []
        for layer in layers[layersIdx[0]:]:
            layer_type = type(layer)
            layer_config = layer.get_config()
            layer_name = layer_config['name']
            split_layer_names.append(layer_name)
            layer_config['name'] = "{}".format(layer_name)
            split_layers.append(layer_type.from_config(layer_config))
        
        back_layer_names = []
        for layer in layers[:layersIdx[0]]:
            layer_type = type(layer)
            layer_config = layer.get_config()
            layer_name = layer_config['name']
            back_layer_names.append(layer_name)

        output_layers_from_backbone = []
        # The input of the branch is tensor with the same shape as the output tensor of the backbone model
        branch_input = []
        branch_dropout = []
        branch_layer = []
        n_layer = layersIdx[0]
        jj = 0
        for i in range(len(split_layers)):
            # Check if we need to add dropout in this layer
            if jj < len(layersIdx) and n_layer == layersIdx[jj]:
                # Get the connections the layer had
                connections = model.get_config()['layers'][n_layer]['inbound_nodes'][0]
                if len(connections) == 1:
                    con_name = connections[0][0]
                    if con_name in split_layer_names:
                        conIdx = split_layer_names.index(con_name)
                        con_name_in_branch = "{}".format(con_name)
                        for bl in branch_layer:
                            if bl.name.split('/')[0] == con_name_in_branch:
                                # Check if the connected layer is conv or dense
                                if "input" not in bl.name:
                                    con_idx = split_layer_names.index(bl.name.split('/')[0].split('_branch')[0])
                                    con_type = type(split_layers[con_idx])
                                else:
                                    con_type = type(tf.keras.layers.Input)

                                branch_dropout.append(custom_layers.InferenceDropout(self.droprate)(bl, training=True))
    
                        # Add the layer in which we apply dropout, connect it to the dropout layer just inserted
                        branch_layer.append(split_layers[i](branch_dropout[-1]))
                    elif con_name in back_layer_names:
                        conIdx = back_layer_names.index(con_name)
                        # Here we need to look for the rest of layers
                        for layer in model.layers[:layersIdx[0]]:
                            layer_name = layer.get_config()['name']
                            if layer_name == con_name:
                                # Create a new input for this model that connects to the layer of the backbone model
                                branch_input.append(tf.keras.layers.Input(shape=layer.output.get_shape()[1:]))
                                output_layers_from_backbone.append(layer.output)
                                branch_dropout.append(custom_layers.InferenceDropout(self.droprate)(branch_input[-1], training=True))
                                break
                        # Add the layer in which we apply dropout, connect it to the dropout layer just inserted
                        branch_layer.append(split_layers[i](branch_dropout[-1]))
                    else:
                        utils.error_and_exit("Connection {} for dropout layer not found".format(con_name))
        
                else:
                    # If this layer depends on two or more inputs, then we need to check which input is from the branch and is
                    # a conv or dense layer. Then add a dropout layer connected to those layers that fulfill the requirement.
    
                    # Get the connections the layer had
                    con_list = []
                    for con in connections:
                        con_name = con[0]
                        if con_name in split_layer_names:
                            conIdx = split_layer_names.index(con_name)
                            con_name_in_branch = "{}".format(con_name)
                            for bl in branch_layer:
                                if bl.name.split('/')[0] == con_name_in_branch:
                                    con_list.append(bl)
                                    break
                        elif con_name in back_layer_names:
                            conIdx = back_layer_names.index(con_name)
                            for layer in model.layers[:layersIdx[0]]:
                                layer_name = layer.get_config()['name']
                                if layer_name == con_name:
                                    # Create a new input for this model that connects to the layer of the backbone model
                                    branch_input.append(tf.keras.layers.Input(shape=layer.output.get_shape()[1:]))
                                    output_layers_from_backbone.append(layer.output)
                                    con_list.append(branch_input[-1])
                                    break
                        else:
                            utils.error_and_exit("Connection {} for layer {} (2+ inputs) not found".format(con_name, n_layer))
    
                    # Now check which of those connections are dense or convolution
                    for idx, con in enumerate(con_list):
                        if "input" not in con.name:
                            con_idx = split_layer_names.index(con.name.split('/')[0].split('_branch')[0])
                            con_type = type(split_layers[con_idx])
                        else:
                            con_type = type(tf.keras.layers.Input)
    
                        # Add a dropout layer and modify the con_list
                        branch_dropout.append(custom_layers.InferenceDropout(self.droprate)(con_list[idx], training=True))
                        con_list[idx] = branch_dropout[-1]
    
                    # Connect the layer
                    branch_layer.append(split_layers[i](con_list))
                jj += 1
        
            else:
                # We do not need to apply dropout in this layer, so just add it, restoring connectivity
                # Get the connections the layer had
                connections = model.get_config()['layers'][n_layer]['inbound_nodes'][0]
        
                # If we only have one connection, just search for the layer and connect to it
                if len(connections) == 1:
                    con_name = connections[0][0]
                    if con_name in split_layer_names:
                        conIdx = split_layer_names.index(con_name)
                        con_name_in_branch = "{}".format(con_name)
                        # Search for the connection in the outputs of the branched model
                        for bl in branch_layer:
                            # If we find the connection, connect the layer
                            if bl.name.split('/')[0] == con_name_in_branch:
                                branch_layer.append(split_layers[i](bl))
                                break
                    elif con_name in back_layer_names:
                        conIDx = back_layer_names.index(con_name)
                        # Here we need to look for the rest of the layers
                        for layer in model.layers[:layersIdx[0]]:
                            layer_name = layer.get_config()['name']
                            if layer_name == con_name:
                                output_layers_from_backbone.append(layer.output)
                                # Create a new input for this model that connects to the layer of the backbone model
                                branch_input.append(tf.keras.layers.Input(shape=layer.output.get_shape()[1:]))
                                branch_layer.append(split_layers[i](branch_input[-1]))
                                break
                    else:
                        utils.error_and_exit("Connection {} for layer {} not found".format(con_name, n_layer))
        
                # If we have 2+ connections, we need to first generate a list of connections
                # and then connect the layer to them
                else:
                    # Get the connections the layer had
                    con_list = []
                    for con in connections:
                        con_name = con[0]
                        if con_name in split_layer_names:
                            conIdx = split_layer_names.index(con_name)
                            con_name_in_branch = "{}".format(con_name)
                            for bl in branch_layer:
                                if bl.name.split('/')[0] == con_name_in_branch:
                                    con_list.append(bl)
                                    break
                        elif con_name in back_layer_names:
                            conIdx = back_layer_names.index(con_name)
                            for layer in model.layers[:layersIdx[0]]:
                                layer_name = layer.get_config()['name']
                                if layer_name == con_name:
                                    # Create a new input for this model that connects to the layer of the backbone model
                                    branch_input.append(tf.keras.layers.Input(shape=layer.output.get_shape()[1:]))
                                    output_layers_from_backbone.append(layer.output)
                                    con_list.append(branch_input[-1])
                                    break
                        else:
                            utils.error_and_exit("Connection {} for layer {} (2+ inputs) not found".format(con_name, n_layer))
        
                    # Connect the layer
                    branch_layer.append(split_layers[i](con_list))
        
            n_layer += 1
        
        mcdoModel = tf.keras.models.Model(inputs=branch_input, outputs=branch_layer[-1], name="mcdo")
        
        # Now that we know the actual outputs we'll need from the backbone model, regenerate it
        #FIXME: need to get only one copy of the output layers from backbone in the main loop
        n_outputs_from_backbone = len(output_layers_from_backbone)
        actual_outputs = output_layers_from_backbone[:n_outputs_from_backbone]
        backboneModel = tf.keras.models.Model(inputs=model.input, outputs=actual_outputs, name='backbone')
    
        # Merge models
        x = backboneModel.output
        y = mcdoModel(x)
        finalModel = tf.keras.Model(backboneModel.input, y, name="{}_mcdo".format(model.name))

        # Print some metrics from the new models
        orig_flops = get_flops(model) / 10 ** 9
        new_flops = get_flops(finalModel) / 10 ** 9
        backbone_flops = get_flops(backboneModel) / 10 ** 9
        mcdo_flops = get_flops(mcdoModel) / 10 ** 9
        print("################################################################################")
        print("# Summary                                                                      #")
        print("################################################################################")
        print("")
        print("* NOTE: Following data considers batch size equal to 1")
        print("")
        print("* INFO: Total number of layers includes any layer in the model (e.g., input and activation layers")
        print("* INFO: OPS refers to expected F32 operations that will execute during 1 forward pass")
        print("* INFO: Backbone is the part of the model were no dropout layers have been inserted")
        print("* INFO: MCDO refers to the part of the models were dropout layers have been inserted")
        print("* INFO: MCDO OPS will be increase linearly with the number of branches created using the `branch` subcommand")
        print("")
        print("- Total number of layers on original model: {}".format(len(model.layers)))
        print("- Number of dropout layers inserted:        {}".format(self.inserted_dropout_layers))
        print("- Original model OPS:                       {:.04} G".format(orig_flops))
        print("- MCDO model OPS:                           {:.04} G".format(new_flops))
        print("    + Backbone OPS:                         {:.04} G".format(backbone_flops))
        print("    + MCDO OPS:                             {:.04} G".format(mcdo_flops))

        return finalModel

