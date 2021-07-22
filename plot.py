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

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import os

# Example input file
"""
corruption,corruption_level,accuracy,brier_score,log_prob,ece
no_corruption,0,0.8211,-0.7458909,-0.5305062,0.0327417109340429
brightness,1,0.8204,-0.74350697,-0.5379851,0.03484476543962957
brightness,2,0.8166,-0.7372193,-0.5531451,0.03741934036910537
brightness,3,0.8087,-0.7275195,-0.5742648,0.04281377467811109
brightness,4,0.7993,-0.7130572,-0.605181,0.04796253250241279
brightness,5,0.7665,-0.6686999,-0.70747876,0.06516728096306322
contrast,1,0.7948,-0.71358025,-0.59739286,0.0336852624177933
contrast,2,0.6724,-0.55007875,-0.9837515,0.07694917090237141
contrast,3,0.5454,-0.38977706,-1.4168482,0.14923991772830486
contrast,4,0.398,-0.19149104,-2.1185203,0.2450916025489569
contrast,5,0.2327,0.006430388,-3.1845338,0.33708992242068053
"""

parser = argparse.ArgumentParser(prog='plot.py')
parser.add_argument('--input', type=str, nargs='+', required=True, help='List of CSV files to process. Input filenames are expected to be in the format <prec>_<model_name>.csv. This plot assumes every file contains data from different models using the same precision')
opt = parser.parse_args()

intensity = []
ece = {}
accuracy = {}
model_name = []
model_prec = []
firstInput = True
for input in opt.input:
    prec, name = os.path.basename(os.path.splitext(input)[0]).split('_', 1)
    model_name.append(name)
    model_prec.append(prec)
    print("Processing model {} with precision {}".format(model_name[-1], model_prec[-1]))
    if firstInput:
        with open(input, 'r') as file:
            firstLine = True
            for line in file.readlines():
                # Skip first line since it contains the header
                if firstLine:
                    firstLine = False
                    continue
                values = line.split(',')
                if model_name[-1] not in ece:
                    ece[model_name[-1]] = []
                if model_name[-1] not in accuracy:
                    accuracy[model_name[-1]] = []
                intensity.append(values[1])
                ece[model_name[-1]].append(float(values[-1]))
                accuracy[model_name[-1]].append(float(values[2]))
        firstInput = False
    else:
        with open(input, 'r') as file:
            firstLine = True
            for line in file.readlines():
                # Skip first line since it contains the header
                if firstLine:
                    firstLine = False
                    continue
                values = line.split(',')
                if model_name[-1] not in ece:
                    ece[model_name[-1]] = []
                if model_name[-1] not in accuracy:
                    accuracy[model_name[-1]] = []
                ece[model_name[-1]].append(float(values[-1]))
                accuracy[model_name[-1]].append(float(values[2]))

# Build a single dictionary with all the importand data in it
dictAcc = { 'Shift intensity': intensity }
for key, value in accuracy.items():
    dictAcc[key] = value

dictEce = { 'Shift intensity': intensity }
for key, value in ece.items():
    dictEce[key] = value

# Output filenames
outputAcc = "{}_acc.pdf".format(model_prec[0])
outputEce = "{}_ece.pdf".format(model_prec[0])

# Plot accuracy data
figAcc, axAcc = plt.subplots()
dfAcc = pd.DataFrame(dictAcc)
dfAcc = dfAcc[dictAcc.keys()]
ddAcc = pd.melt(dfAcc, id_vars=list(dfAcc)[0], value_vars=list(dfAcc)[1:], value_name='Accuracy', var_name='model')
bA = sns.boxplot(x='Shift intensity',y='Accuracy',data=ddAcc,hue='model', ax=axAcc)

# Put the legend outside
plt.legend(loc='center right', bbox_to_anchor=(1.3, 0.5), ncol=1, fancybox=True, fontsize='small')


# Plot ECE data
figEce, axEce = plt.subplots()
dfEce = pd.DataFrame(dictEce)
dfEce = dfEce[dictEce.keys()]
ddEce = pd.melt(dfEce, id_vars=list(dfEce)[0], value_vars=list(dfEce)[1:], value_name='ECE', var_name='model')
bE = sns.boxplot(x='Shift intensity',y='ECE',data=ddEce,hue='model', ax=axEce)
hE, lE = bE.get_legend_handles_labels()

# Put the legend outside
plt.legend(loc='center right', bbox_to_anchor=(1.3, 0.5), ncol=1, fancybox=True, fontsize='small')

# Put a bit more space between subplots
figAcc.tight_layout()
figEce.tight_layout()

# Save the graph
figAcc.savefig(outputAcc)
figEce.savefig(outputEce)
