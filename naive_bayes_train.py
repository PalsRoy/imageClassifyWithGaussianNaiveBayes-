"""
Standalone Gaussian Naive Bayes - training only, for now.
We'll fold this into the class once it works.
"""

import numpy as np
import cv2
from feature_extraction import extract_features


# A tiny number we add to all variances. Why?
# If a class happens to have nearly-zero variance on some feature
# (e.g. all training pixels of that class have the exact same value),
# the Gaussian formula would divide by zero and explode. Adding a small
# constant prevents that. This is sometimes called "variance smoothing".
VARIANCE_SMOOTHING = 1e-3


def fit_gaussian_naive_bayes(features, labels, classes):
    """
    Compute per-class, per-feature means and variances.

    Inputs
    ------
    features : (N, F) array of pixel features
    labels   : (N,)   array of class labels (each in `classes`)
    classes  : (K,)   array of class IDs we want to model

    Returns
    -------
    means     : (K, F) array - means[k, j] = mean of feature j for class k
    variances : (K, F) array - variances[k, j] = variance of feature j for class k

    Notes
    -----
    This is the entire "training" step. There's no iteration, no learning rate,
    no gradient descent - we just compute statistics directly from the data.
    """
    K = len(classes)
    F = features.shape[1]

    means = np.zeros((K, F), dtype=np.float64)
    variances = np.zeros((K, F), dtype=np.float64)

    for i, class_id in enumerate(classes):
        # Boolean mask: True for every pixel belonging to this class
        mask = (labels == class_id)
        # Pull out only those pixels' features
        class_features = features[mask]
        # Compute mean across the *samples axis* (axis=0). The result is
        # a length-F vector: one mean per feature.
        means[i] = class_features.mean(axis=0)
        # Same for variance. Add the smoothing term to keep things numerically
        # stable on pathologically narrow classes.
        variances[i] = class_features.var(axis=0) + VARIANCE_SMOOTHING

    return means, variances


# ----------------------------------------------------------------------------
# Self-test - train on the real data and inspect what we learnt
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    # Load training image and mask
    img = cv2.imread('./data/training_image.jpg')
    if img is None:
        raise FileNotFoundError("training_image.jpg not found in working dir")
    mask = cv2.imread('./data/training_mask.png')
    if mask is None:
        raise FileNotFoundError("training_mask.png not found in working dir")
    mask = mask[:, :, 0]  # all 3 channels are identical, take just one

    # Extract features
    print("Extracting features...")
    features, H, W = extract_features(img)
    labels = mask.flatten()

    # Filter out the spurious labels (0, 6, 7, 8) we identified earlier.
    # Only keep pixels with valid class labels 1..5.
    valid = (labels >= 1) & (labels <= 5)
    features_valid = features[valid]
    labels_valid = labels[valid]
    print(f"Training on {len(labels_valid):,} valid pixels "
          f"(filtered out {(~valid).sum():,} spurious-label pixels)")

    # Train
    classes = np.array([1, 2, 3, 4, 5])
    means, variances = fit_gaussian_naive_bayes(features_valid, labels_valid, classes)

    # Inspect what the model learnt
    feature_names = ['R', 'G', 'B', 'H', 'S', 'V', 'ExG', 'std']
    class_names = {1: 'Building', 2: 'Road', 3: 'Tree', 4: 'Vehicle', 5: 'Grass'}

    print("\n=== Learnt MEANS (mu_kj) ===")
    print(f"{'Class':<10s} | " + " ".join(f"{n:>7s}" for n in feature_names))
    for i, k in enumerate(classes):
        print(f"{class_names[k]:<10s} | " +
              " ".join(f"{m:>7.1f}" for m in means[i]))

    print("\n=== Learnt VARIANCES (sigma^2_kj) ===")
    print(f"{'Class':<10s} | " + " ".join(f"{n:>7s}" for n in feature_names))
    for i, k in enumerate(classes):
        print(f"{class_names[k]:<10s} | " +
              " ".join(f"{v:>7.0f}" for v in variances[i]))

    print(f"\nModel size: {means.size + variances.size} numbers total "
          f"({means.size} means + {variances.size} variances)")