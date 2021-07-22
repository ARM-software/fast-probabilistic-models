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

import sys
import os
import numpy as np
from typing import List
import scipy.special as sc
import six
from six.moves import range

def error_and_exit(msg: str):
    print('[ERROR]: ', msg)
    sys.exit(1)

def calculate_output(output_arr: np.ndarray, confidence_interval: float = 0.95, non_parametric: bool = True):
    """
      It calculates an averaged output across MCDO samples with confidence intervals for each outputted class.

    Parameters
    ----------
    output_arr: np.ndarray of shape (num_branches, num_elems, output_size)
        An array with the outputs of each MCDO sample. Each MCDO sample is expected to NOT have softmax
        applied (i.e., accepts logits).
    confidence_interval: float
        The confidence interval between [0, 1]. If non_parametric=False then only the values 0.8, 0.9, 0.95, and 0.99 are
        currently available.
    non_parametric: bool, default=True
        If non_parametric=True, calculates non parametric confidence intervals, in which no assumptions are made about
        the underlying distribution and intervals are calculated based only on the percentiles.
        If non_parametric=False, intervals are calculated assuming samples are normally distributed, and only the
        confidence intervals 0.8, 0.9, 0.95, and 0.99 can be used in this case. Be careful as the assumption of normality
        can generate unreasonable numbers when num_elems is too small (e.g. values below zero when output is softmaxed).

    Returns
    -------
    mean_preds : ndarray of shape (num_elems, output_size)
        Averaged scores across sampled outputs
    lower_lim : ndarray of shape (num_elems, output_size)
        Lower limit of the confidence interval per class output
    upper_lim : ndarray of shape (num_elems, output_size)
        Upper limit of the confidence interval per class output
    std_preds : ndarray of shape (num_elems, output_size)
        Standard deviations across sampled outputs
    """
    if confidence_interval < 0 or confidence_interval > 1:
        error_and_exit('Confidence interval needs to be between 0 and 1.')
    if output_arr.ndim != 3:
        error_and_exit('output_arr does not have the expected dimension of [num_branches, num_elems, output_size].')

    # Softmax the logits (last axis) 
    predictions = sc.softmax(output_arr, axis=-1)

    # [num_elems, 10]
    mean_preds = np.mean(predictions, axis=0)
    std_preds = np.std(predictions, axis=0)

    if non_parametric:
        ci = confidence_interval
        lower_lim = np.quantile(predictions, 0.5 - ci / 2, axis=0)  # lower limit of the CI
        upper_lim = np.quantile(predictions, 0.5 + ci / 2, axis=0) # upper limit of the CI
    else:
        num_samples = predictions.shape[1]
        zscores = {0.8: 1.282, 0.9: 1.645, 0.95: 1.96, 0.99: 2.576}

        if round(confidence_interval, 2) not in zscores.keys():
            error_and_exit(
                f'Confidence interval not supportted. Only the following are supported for parametric calculation: {zscores.keys()}')
        if num_samples < 30:
            print(
                'Warning: calculating a parametric confidence interval with number of samples < 30 is not recommended.')

        z = zscores[round(confidence_interval, 2)]

        se = std_preds / np.sqrt(num_samples)
        lower_lim = mean_preds - z * se  # lower limit of the CI
        upper_lim = mean_preds + z * se  # upper limit of the CI

    return mean_preds, lower_lim, upper_lim, std_preds

################################################################################
# Inference metrics                                                            #
################################################################################
def bin_centers_of_mass(probabilities, bin_edges):
  probabilities = np.where(probabilities == 0, 1e-8, probabilities)
  indices = np.digitize(probabilities, bin_edges, right=True)
  return np.array([np.mean(probabilities[indices == i])
                   for i in range(1, len(bin_edges))])

def bin_predictions_and_accuracies(probabilities, ground_truth, bins=10):
  """A helper function which histograms a vector of probabilities into bins.

  Args:
    probabilities: A numpy vector of N probabilities assigned to each prediction
    ground_truth: A numpy vector of N ground truth labels in {0,1}
    bins: Number of equal width bins to bin predictions into in [0, 1], or an
      array representing bin edges.

  Returns:
    bin_edges: Numpy vector of floats containing the edges of the bins
      (including leftmost and rightmost).
    accuracies: Numpy vector of floats for the average accuracy of the
      predictions in each bin.
    counts: Numpy vector of ints containing the number of examples per bin.
  """
  _validate_probabilities(probabilities)
  _check_rank_nonempty(rank=1,
                       probabilities=probabilities,
                       ground_truth=ground_truth)

  if len(probabilities) != len(ground_truth):
    raise ValueError(
        'Probabilies and ground truth must have the same number of elements.')

  if [v for v in ground_truth if v not in [0., 1., True, False]]:
    raise ValueError(
        'Ground truth must contain binary labels {0,1} or {False, True}.')

  if isinstance(bins, int):
    num_bins = bins
  else:
    num_bins = bins.size - 1

  # Ensure probabilities are never 0, since the bins in np.digitize are open on
  # one side.
  probabilities = np.where(probabilities == 0, 1e-8, probabilities)
  counts, bin_edges = np.histogram(probabilities, bins=bins, range=[0., 1.])
  indices = np.digitize(probabilities, bin_edges, right=True)
  accuracies = np.array([np.mean(ground_truth[indices == i])
                         for i in range(1, num_bins + 1)])
  return bin_edges, accuracies, counts

def expected_calibration_error(probabilities, ground_truth, bins=15):
  """Compute the expected calibration error of a set of preditions in [0, 1].

  Args:
    probabilities: A numpy vector of N probabilities assigned to each prediction
    ground_truth: A numpy vector of N ground truth labels in {0,1, True, False}
    bins: Number of equal width bins to bin predictions into in [0, 1], or
      an array representing bin edges.
  Returns:
    Float: the expected calibration error.
  """
  bin_edges, accuracies, counts = bin_predictions_and_accuracies(
      probabilities, ground_truth, bins)
  bin_centers = bin_centers_of_mass(probabilities, bin_edges)
  num_examples = np.sum(counts)

  ece = np.sum([(counts[i] / float(num_examples)) * np.sum(
      np.abs(bin_centers[i] - accuracies[i]))
                for i in range(bin_centers.size) if counts[i] > 0])
  return ece

def _check_rank_nonempty(rank, **kwargs):
  for key, array in six.iteritems(kwargs):
    if len(array) <= 1 or array.ndim != rank:
      raise ValueError(
          '%s must be a rank-1 array of length > 1; actual shape is %s.' %
          (key, array.shape))

def _validate_probabilities(probabilities, multiclass=False):
  if np.max(probabilities) > 1. or np.min(probabilities) < 0.:
    raise ValueError('All probabilities must be in [0,1].')
  if multiclass and not np.allclose(1, np.sum(probabilities, axis=-1),
                                    atol=1e-5):
    raise ValueError(
        'Multiclass probabilities must sum to 1 along the last dimension.')

def get_multiclass_predictions_and_correctness(probabilities, labels, top_k=1):
  """Returns predicted class, correctness boolean vector."""
  _validate_probabilities(probabilities, multiclass=True)
  _check_rank_nonempty(rank=1, labels=labels)
  _check_rank_nonempty(rank=2, probabilities=probabilities)

  if top_k == 1:
    class_predictions = np.argmax(probabilities, -1)
    top_k_probs = probabilities[np.arange(len(labels)), class_predictions]
    is_correct = np.equal(class_predictions, labels)
  else:
    top_k_probs, is_correct = _filter_top_k(probabilities, labels, top_k)

  return top_k_probs, is_correct

def expected_calibration_error_multiclass(probabilities, labels, bins=15,
                                          top_k=1):
  """Computes expected calibration error from Guo et al. 2017.

  For details, see https://arxiv.org/abs/1706.04599.
  Note: If top_k is None, this only measures calibration of the argmax
    prediction.

  Args:
    probabilities: Array of probabilities of shape [num_samples, num_classes].
    labels: Integer array labels of shape [num_samples].
    bins: Number of equal width bins to bin predictions into in [0, 1], or
      an array representing bin edges.
    top_k: Integer or None. If integer, use the top k predicted
      probabilities in ECE calculation (can be informative for problems with
      many classes and lower top-1 accuracy). If None, use all classes.
  Returns:
    float: Expected calibration error.
  """

  top_k_probs, is_correct = get_multiclass_predictions_and_correctness(
      probabilities, labels, top_k)
  top_k_probs = top_k_probs.flatten()
  is_correct = is_correct.flatten()
  return expected_calibration_error(top_k_probs, is_correct, bins)

def metrics_from_stats(stats):
  """Compute metrics from a stats dictionary."""
  labels, probs = stats['labels'], stats['probs']

  # Reshape binary predictions to 2-class.
  if len(probs.shape) == 1:
    probs = np.stack([1-probs, probs], axis=-1)
  assert len(probs.shape) == 2

  predictions = np.argmax(probs, axis=-1)
  accuracy = np.equal(labels, predictions)

  label_probs = probs[np.arange(len(labels)), labels]
  log_probs = np.maximum(-1e10, np.log(label_probs))
  brier_scores = np.square(probs).sum(-1) - 2 * label_probs
  ece = expected_calibration_error_multiclass(probs, labels)
    

  return {'accuracy': accuracy.mean(0),
          'brier_score': brier_scores.mean(0),
          'log_prob': log_probs.mean(0),
          'ece': ece}

