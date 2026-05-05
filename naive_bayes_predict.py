# -*- coding: utf-8 -*-
"""
Standalone Gaussian Naive Bayes - PREDICTION step.
Loads the model trained earlier and applies it to a new image.
"""

import numpy as np
import cv2
import sys
sys.path.insert(0, '.')
from feature_extraction import extract_features
from naive_bayes_train import fit_gaussian_naive_bayes, VARIANCE_SMOOTHING


def predict_gaussian_naive_bayes(features, means, variances, classes,
                                 log_prior, chunk_size=500_000):
    """
    Predict a class label for every pixel.

    We process pixels in CHUNKS to control memory usage. With 12M pixels
    a single (N, F) intermediate array of float64 would be ~770 MB; doing
    it in chunks of 500k means each intermediate is ~30 MB. Same answer,
    much less RAM.

    Inputs
    ------
    features   : (N, F) array - one row per pixel
    means      : (K, F) array - mu_kj
    variances  : (K, F) array - sigma^2_kj
    classes    : (K,)   array - the actual class IDs (e.g. [1,2,3,4,5])
    log_prior  : (K,)   array - log P(class_k), one per class
    chunk_size : int - how many pixels to process at a time

    Returns
    -------
    predictions : (N,) array of class IDs

    The maths
    ---------
    For each pixel n and class k we want the log-score:

        score[n, k] = log_prior[k]
                    + sum over features j of log_N(f_nj ; mu_kj, sigma^2_kj)

    where the per-feature log-Gaussian is:

        log_N(f, mu, var) = -0.5 * log(2*pi*var) - (f - mu)^2 / (2*var)
    """
    K = len(classes)
    N, F = features.shape

    # Precompute the constant term per (class, feature) - doesn't depend
    # on pixel value. Shape (K, F). Calculating this once saves work.
    log_norm_const = -0.5 * np.log(2.0 * np.pi * variances)  # (K, F)

    # Output array. int16 is plenty since labels are 1..5.
    predictions = np.empty(N, dtype=np.int16)

    # Process pixels in chunks
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        feats_chunk = features[start:end].astype(np.float64)  # (chunk, F)

        # Compute log-posterior for this chunk: shape (chunk, K)
        log_post = np.empty((end - start, K), dtype=np.float64)

        for k in range(K):
            mu_k = means[k]      # (F,)
            var_k = variances[k] # (F,)

            # Broadcasting: (chunk, F) - (F,) -> (chunk, F)
            diff_sq = (feats_chunk - mu_k) ** 2
            # Per-feature log-Gaussian, then sum over features
            log_gauss_per_feat = log_norm_const[k] - diff_sq / (2.0 * var_k)
            log_likelihood_k = log_gauss_per_feat.sum(axis=1)  # (chunk,)

            log_post[:, k] = log_likelihood_k + log_prior[k]

        # argmax over classes for this chunk
        best_idx = np.argmax(log_post, axis=1)        # (chunk,)
        predictions[start:end] = classes[best_idx]

    return predictions


# ----------------------------------------------------------------------------
# End-to-end test: train on training image, predict on training image,
# and check how often we agree with the ground-truth mask.
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    # --- Train ---
    img = cv2.imread('./data/training_image.jpg')
    mask = cv2.imread('./data/training_mask.png')[:, :, 0]
    print("Extracting training features...")
    features, H, W = extract_features(img)
    labels = mask.flatten()
    valid = (labels >= 1) & (labels <= 5)

    classes = np.array([1, 2, 3, 4, 5])
    print("Training...")
    means, variances = fit_gaussian_naive_bayes(
        features[valid], labels[valid], classes
    )

    # Uniform prior - log(1/5) for every class.
    log_prior = np.log(np.full(len(classes), 1.0 / len(classes)))

    # --- Predict on the training image (in-sample sanity check) ---
    print("Predicting on training image...")
    predictions = predict_gaussian_naive_bayes(
        features, means, variances, classes, log_prior
    )

    # --- Compute balanced accuracy on the labelled (valid) pixels ---
    # Balanced accuracy = mean of per-class recalls.
    print("\n=== In-sample performance (training image) ===")
    class_names = {1: 'Building', 2: 'Road', 3: 'Tree', 4: 'Vehicle', 5: 'Grass'}
    recalls = []
    for k in classes:
        truth_k_mask = (labels == k)
        if truth_k_mask.sum() == 0:
            continue
        # Of all the pixels truly belonging to class k, how many did we
        # correctly predict as k?
        true_positives = ((predictions == k) & truth_k_mask).sum()
        recall_k = true_positives / truth_k_mask.sum()
        recalls.append(recall_k)
        print(f"  Recall[{class_names[k]:<8s}] = {recall_k*100:5.1f}%   "
              f"(TP={true_positives:>10,} / total={truth_k_mask.sum():>10,})")

    bal_acc = np.mean(recalls)
    print(f"\n  >>> Balanced accuracy: {bal_acc*100:.2f}% <<<")