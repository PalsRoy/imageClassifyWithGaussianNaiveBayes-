# -*- coding: utf-8 -*-
"""
Standalone feature extraction module - we will fold this into the class later.
For now we keep it separate so we can test it and inspect outputs.
"""

import numpy as np
import cv2


def extract_features(image_bgr, std_window_size=7):
    """
    Compute an 8-dimensional feature vector for every pixel in the image.

    Inputs
    ------
    image_bgr : numpy array of shape (H, W, 3), dtype uint8
        The image as read by cv2.imread (channels are B, G, R in that order).
    std_window_size : int
        Side length of the square window used for the local-std texture feature.
        Must be odd. 7 is a sensible default - small enough to be fast, large
        enough to average over local leaf/grass texture.

    Returns
    -------
    features : numpy array of shape (H*W, 8), dtype float32
        Each row is one pixel's feature vector, in this order:
            [R, G, B, H, S, V, ExG, local_std]
        Pixels are flattened in row-major order so that row r, column c maps to
        index r*W + c, which is what numpy.reshape uses by default.
    H, W : ints
        Original image dimensions, returned so the caller can reshape predictions
        back into a 2D mask later on.
    """
    H_img, W_img = image_bgr.shape[:2]

    # --- Colour features in BGR ----------------------------------------------
    # Convert to float32 once so we don't have to keep casting later.
    img_f = image_bgr.astype(np.float32)
    B = img_f[:, :, 0]
    G = img_f[:, :, 1]
    R = img_f[:, :, 2]

    # --- Colour features in HSV ----------------------------------------------
    # OpenCV's HSV: H in [0,179], S and V in [0,255] for uint8 images.
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    H_chan = hsv[:, :, 0]
    S_chan = hsv[:, :, 1]
    V_chan = hsv[:, :, 2]

    # --- Excess Green vegetation index ---------------------------------------
    # ExG is large and positive for green pixels, near zero for grey pixels,
    # and negative for red/brown/blue-dominant pixels. This is the cheapest
    # vegetation detector in remote sensing.
    ExG = 2.0 * G - R - B

    # --- Local standard deviation of the V (brightness) channel --------------
    # Trick: variance = E[X^2] - E[X]^2. We get the local mean of X and of X^2
    # using box blur (an O(1)-per-pixel running sum), then combine. This is
    # much faster than looping pixel-by-pixel in Python.
    # The np.maximum(..., 0) clamps tiny negative numbers from float rounding
    # back to zero so we can take a square root safely.
    mean_V = cv2.blur(V_chan, (std_window_size, std_window_size))
    mean_V_squared = cv2.blur(V_chan ** 2, (std_window_size, std_window_size))
    local_variance = np.maximum(mean_V_squared - mean_V ** 2, 0.0)
    local_std = np.sqrt(local_variance)

    # --- Stack all features into a single (H, W, 8) tensor -------------------
    # Order chosen for readability when debugging:
    feature_stack = np.stack([R, G, B, H_chan, S_chan, V_chan, ExG, local_std],
                             axis=-1).astype(np.float32)

    # --- Flatten so each row is one pixel -------------------------------------
    # reshape(-1, 8) collapses the H and W axes into one, preserving channel.
    features = feature_stack.reshape(-1, 8)

    return features, H_img, W_img


# ------------------------------------------------------------------------
if __name__ == "__main__":
    img = cv2.imread('./data/training_image.jpg')
    print(f"Image shape: {img.shape}")

    feats, H, W = extract_features(img)
    print(f"Feature matrix shape: {feats.shape}")
    print(f"Feature matrix dtype: {feats.dtype}")
    print(f"Memory footprint: {feats.nbytes / (1024**2):.1f} MB")

    # Print min/max for each feature so we can sanity-check the ranges
    feature_names = ['R', 'G', 'B', 'H', 'S', 'V', 'ExG', 'local_std']
    print("\nFeature ranges across all 12M pixels:")
    for i, name in enumerate(feature_names):
        col = feats[:, i]
        print(f"  {name:>9s}: min={col.min():>8.2f}  "
              f"max={col.max():>8.2f}  "
              f"mean={col.mean():>8.2f}  "
              f"std={col.std():>8.2f}")