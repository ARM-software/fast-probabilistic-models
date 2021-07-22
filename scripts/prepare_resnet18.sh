#!/bin/bash

################################################################################
# SPDX-License-Identifier: Apache-2.0                                          #
#                                                                              #
# Copyright (C) 2021, Arm Limited and contributors                             #
#                                                                              #
# Licensed under the Apache License, Version 2.0 (the "License");              #
# you may not use this file except in compliance with the License.             #
# You may obtain a copy of the License at                                      #
#                                                                              #
#     http://www.apache.org/licenses/LICENSE-2.0                               #
#                                                                              #
# Unless required by applicable law or agreed to in writing, software          #
# distributed under the License is distributed on an "AS IS" BASIS,            #
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.     #
# See the License for the specific language governing permissions and          #
# limitations under the License.                                               #
################################################################################

tf_model_path='./experiment_models/tf'
tflite_model_path='./experiment_models/tflite'

mkdir -p ${tf_model_path} ${tflite_model_path}

# STEP 1: Build the vanilla model
echo "[`date +%T`] Building vanilla ResNet18 model"
python ResNet18.py build --save_to ${tf_model_path} \
                         --save_filename vanilla \
                         --save_format h5

# STEP 2: Create MCDO variants
for mcdo_kind in ll_mcdo partial_mcdo full_mcdo
do
    if [ "$mcdo_kind" == "full_mcdo" ]
    then
        starting_layer='conv2d_1'
        droprate=0.054988
    elif [ "$mcdo_kind" == "ll_mcdo" ]
    then
        starting_layer='dense'
        droprate=0.319811
    elif [ "$mcdo_kind" == "partial_mcdo" ]
    then
        starting_layer='conv2d_10'
        #TODO: find the best droprate for this MCDO variant
        droprate=0.054988
    fi
    echo "[`date +%T`] Generating MCDO variant ${mcdo_kind} of ResNet18 with a droprate of ${droprate}"
    python ResNet18.py mcdo --model ${tf_model_path}/vanilla.h5 \
        --save_to ${tf_model_path} \
        --save_filename ${mcdo_kind} \
        --save_format h5 \
        --starting_layer ${starting_layer} \
        --droprate ${droprate}
done

# STEP 3: Train the models
for kind in vanilla ll_mcdo partial_mcdo full_mcdo
do
    if [ "$kind" == "ll_mcdo" ]
    then
        modelFilename="${kind}"
        batchSize=16
        initialLearningRate=0.000313
    elif [ "$kind" == "partial_mcdo" ]
    then
        modelFilename="${kind}"
        #TODO: Find best batch and initial lr
        batchSize=7
        initialLearningRate=0.000250
    elif [ "$kind" == "full_mcdo" ]
    then
        modelFilename="${kind}"
        batchSize=5
        initialLearningRate=0.000250
    elif [ "$kind" == "vanilla" ]
    then
        modelFilename="${kind}"
        batchSize=7
        initialLearningRate=0.000717
    fi

    echo "[`date +%T`] Training ${modelFilename} model"
    python ResNet18.py train --model ${tf_model_path}/${modelFilename}.h5 \
                             --save_to ${tf_model_path} \
                             --save_filename ${modelFilename} \
                             --save_format h5 \
                             --epochs 200 \
                             --batch ${batchSize} \
                             --initial_learning_rate ${initialLearningRate} \
                             --tensorboard_logdir ./tensorboard_logs/${modelFilename}
done

# STEP 4: Branch the models
for kind in ll_mcdo partial_mcdo full_mcdo
do
    modelFilename="${kind}"
    echo "[`date +%T`] Branching ${modelFilename} model"
    python ResNet18.py branch --model ${tf_model_path}/${modelFilename}.h5 \
        --save_to ${tf_model_path} \
        --save_filename ${modelFilename} \
        --save_format h5 \
        --n_branches 5
done

# STEP 5: Convert to TFLite in both fp32 and int8
for model in `ls -tr ${tf_model_path}`
do
    model_name=`basename -s .h5 ${model}`
    for prec in fp32 int8
    do
        echo "[`date +%T`] Generating TFLite model from ${model_name} using ${prec^^} precision"
        if [ "$prec" == "fp32" ]
        then
            python ResNet18.py convert --model ${tf_model_path}/${model} \
                                       --save_to ${tflite_model_path} \
                                       --save_filename ${prec}_${model_name}
        else
            python ResNet18.py convert --model ${tf_model_path}/${model} \
                                       --save_to ${tflite_model_path} \
                                       --save_filename ${prec}_${model_name} \
                                       --int8
        fi
    done
done

# STEP 6: Run inference... but use another script
echo "All models have been generated, you can now use scripts/inference_resnet18.sh"
