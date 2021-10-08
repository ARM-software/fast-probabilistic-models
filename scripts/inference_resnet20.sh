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

runInferenceCorrupted() {
    python ResNet20.py inference --model ${1} --batch ${2} --corruption ${3} --corruption_level ${4} | grep -A1 "accuracy" | tail -n1 >> ${5}
}
runInference() {
    python ResNet20.py inference --model ${1} --batch ${2} | grep -A1 "accuracy" | tail -n1 >> ${3}
}
export -f runInferenceCorrupted
export -f runInference

TF_OR_TFLITE=tflite
if [ $# -ge 1 ]
then
    if [ "$1" == "tf" -o "$1" == "tflite" ]
    then
        TF_OR_TFLITE=$1
    else
        echo "`basename $0` [tf|tflite]"
        exit 0
    fi
fi

if [ "$TF_OR_TFLITE" == "tf" ]
then
    extension="h5"
    fp32_models=(./experiment_models/${TF_OR_TFLITE}/vanilla.${extension}
                 ./experiment_models/${TF_OR_TFLITE}/ll_mcdo.${extension}
                 ./experiment_models/${TF_OR_TFLITE}/partial_mcdo.${extension}
                 ./experiment_models/${TF_OR_TFLITE}/full_mcdo.${extension})
else
    extension="tflite"
    int8_models=(./experiment_models/${TF_OR_TFLITE}/int8_vanilla.${extension}
                 ./experiment_models/${TF_OR_TFLITE}/int8_ll_mcdo.${extension}
                 ./experiment_models/${TF_OR_TFLITE}/int8_partial_mcdo.${extension}
                 ./experiment_models/${TF_OR_TFLITE}/int8_full_mcdo.${extension})
    fp32_models=(./experiment_models/${TF_OR_TFLITE}/fp32_vanilla.${extension}
                 ./experiment_models/${TF_OR_TFLITE}/fp32_ll_mcdo.${extension}
                 ./experiment_models/${TF_OR_TFLITE}/fp32_partial_mcdo.${extension}
                 ./experiment_models/${TF_OR_TFLITE}/fp32_full_mcdo.${extension})
fi


csv_folder="csvs_${TF_OR_TFLITE}"

corruption=(brightness
            contrast
            defocus_blur
            elastic
            fog
            frost
            frosted_glass_blur
            gaussian_blur
            gaussian_noise
            impulse_noise
            jpeg_compression
            motion_blur
            pixelate
            saturate
            shot_noise
            snow
            spatter
            speckle_noise
            zoom_blur)

# This batch size has been found to be fastest after some experimentation
batch_size=50

mkdir -p ${csv_folder}
# FP32
for model in ${fp32_models[@]}
do
    modelName=`basename -s .${extension} $model`
    outputFile=${csv_folder}/${modelName}.csv

    # Print the header into the file, removing potential old stuff in it at the same time
    echo "corruption,corruption_level,accuracy,brier_score,log_prob,ece" > $outputFile

    # No corruption
    echo "[`date +%T`] Evaluating model ${modelName} with vanilla CIFAR-10"
    runInference ${model} ${batch_size} ${outputFile}

    # Corruption
    for cor in ${corruption[@]}
    do
        for cor_level in {1..5}
        do
            echo "[`date +%T`] Evaluating model ${modelName} with corrupted CIFAR-10 (${cor}_${cor_level})"
            runInferenceCorrupted ${model} ${batch_size} ${cor} ${cor_level} ${outputFile}
        done
    done
done

if [ "${TF_OR_TFLITE}" == "tflite" ]
then
    # INT8
    for model in ${int8_models[@]}
    do
        modelName=`basename -s .${extension} $model`
        outputFile=${csv_folder}/${modelName}.csv
    
        # Print the header into the file, removing potential old stuff in it at the same time
        echo "corruption,corruption_level,accuracy,brier_score,log_prob,ece" > $outputFile
    
        # No corruption
        echo "[`date +%T`] Evaluating model ${modelName} with vanilla CIFAR-10"
        runInference ${model} ${batch_size} ${outputFile}
    
        # Corruption
        for cor in ${corruption[@]}
        do
            for cor_level in {1..5}
            do
                echo "[`date +%T`] Evaluating model ${modelName} with corrupted CIFAR-10 (${cor}_${cor_level})"
                runInferenceCorrupted ${model} ${batch_size} ${cor} ${cor_level} ${outputFile}
            done
        done
    done
fi

# Put correct names on CSV files if using TF models
if [ "${TF_OR_TFLITE}" == "tf" ]
then
    cd csvs_${TF_OR_TFLITE}
    for i in `ls`
    do
        mv ${i} fp32_${i}
    done
    cd ..
fi

# Plot for FP32 models
python plot.py --input ./csvs_${TF_OR_TFLITE}/fp32_vanilla.csv \
                       ./csvs_${TF_OR_TFLITE}/fp32_full_mcdo.csv \
                       ./csvs_${TF_OR_TFLITE}/fp32_partial_mcdo.csv \
                       ./csvs_${TF_OR_TFLITE}/fp32_ll_mcdo.csv

# Plot for INT8 models (only for TFLite)
if [ "${TF_OR_TFLITE}" == "tflite" ]
then
    python plot.py --input ./csvs_${TF_OR_TFLITE}/int8_vanilla.csv \
                           ./csvs_${TF_OR_TFLITE}/int8_full_mcdo.csv \
                           ./csvs_${TF_OR_TFLITE}/int8_partial_mcdo.csv \
                           ./csvs_${TF_OR_TFLITE}/int8_ll_mcdo.csv
fi

exit 0
