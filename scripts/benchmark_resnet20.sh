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

TF_OR_TFLITE=tflite
if [ $# -eq 1 ]
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
    batches=( 1 2 4 8 16 32 64 128 )
else
    extension="tflite"
    fp32_models=(./experiment_models/${TF_OR_TFLITE}/fp32_vanilla.${extension}
                 ./experiment_models/${TF_OR_TFLITE}/fp32_ll_mcdo.${extension}
                 ./experiment_models/${TF_OR_TFLITE}/fp32_partial_mcdo.${extension}
                 ./experiment_models/${TF_OR_TFLITE}/fp32_full_mcdo.${extension})

    int8_models=(./experiment_models/${TF_OR_TFLITE}/int8_vanilla.${extension}
                 ./experiment_models/${TF_OR_TFLITE}/int8_ll_mcdo.${extension}
                 ./experiment_models/${TF_OR_TFLITE}/int8_partial_mcdo.${extension}
                 ./experiment_models/${TF_OR_TFLITE}/int8_full_mcdo.${extension})
    batches=( 1 2 4 8 16 40 80 100 )
fi

csv_folder="csvs_benchmark"

mkdir -p ${csv_folder}
# FP32
outputFile=${csv_folder}/fp32_${TF_OR_TFLITE}.csv
# Print the header into the file, removing potential old stuff in it at the same time
echo "model_name,batch_size,model_size[MB],mean[sec],median[sec],max[sec],min[sec],stdev" > $outputFile
for model in ${fp32_models[@]}
do
    for batch_size in ${batches[@]}
    do
        modelName=`basename -s .${extension} $model`
        echo "[`date +%T`] Benchmarking model ${modelName} with batch size ${batch_size}"
        metrics=`python ResNet20.py benchmark --model $model --batch ${batch_size} --repeats 100 --no-header`
        echo "${modelName},${batch_size},${metrics}" >> ${outputFile}
    done
done

if [ "${TF_OR_TFLITE}" == "tflite" ]
then
    # INT8
    outputFile=${csv_folder}/int8_${TF_OR_TFLITE}.csv
    # Print the header into the file, removing potential old stuff in it at the same time
    echo "model_name,batch_size,model_size[MB],mean[sec],median[sec],max[sec],min[sec],stdev" > $outputFile
    for model in ${int8_models[@]}
    do
        for batch_size in ${batches[@]}
        do
            modelName=`basename -s .${extension} $model`
            echo "[`date +%T`] Benchmarking model ${modelName} with batch size ${batch_size}"
            metrics=`python ResNet20.py benchmark --model $model --batch ${batch_size} --repeats 100 --no-header`
            echo "${modelName},${batch_size},${metrics}" >> ${outputFile}
        done
    done
fi

exit 0
