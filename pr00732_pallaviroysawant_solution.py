# -*- coding: utf-8 -*-
"""
EEEM005 Coursework Assignment Answer Sheet.

Aerial image semantic segmentation using a Gaussian Naive Bayes classifier
trained on hand-crafted per-pixel features. Implemented using only NumPy,
SciPy and OpenCV (the latter for I/O and colour-space conversion only).

Author : Pallavi Roy Sawant
ID     : PR00732

Classes:
    1 = Building, 2 = Road, 3 = Tree, 4 = Vehicle, 5 = Grass
"""

# %% Imports
# -----------------------------------------------------------------------------
# - numpy / scipy : explicitly listed as preferred libraries by the assignment.
# - cv2           : permitted for image read/write/colour-space conversion.
# No high-level ML libraries (sklearn, tensorflow, pytorch, ...) are used.
# -----------------------------------------------------------------------------
import numpy as np
import cv2
from scipy.ndimage import median_filter


# %% The model class
# -----------------------------------------------------------------------------
# The structure follows EEEM005AnswerSheetTemplate.py exactly:
#   - __init__ takes no inputs
#   - model_training takes (Training_Image_Name, Training_Image_Mask_Name)
#     and returns nothing
#   - model_testing  takes (Testing_Image_Name, Testing_Image_Mask_Name)
#     and saves the predicted mask to disk and there are no return values.
# Helper methods are private and prefixed with _ to keep the public surfacec.
# -----------------------------------------------------------------------------
class EEEM005_Coursework_Solution:
    """Pixel-wise aerial-image segmenter using Gaussian Naive Bayes."""

    # Hyperparameters ---------------------------------------------------------
    # Used class-level constants so they are easy to find and adjust.
    STD_WINDOW_SIZE       = 7      # window for the local-std texture feature
    MEDIAN_FILTER_SIZE    = 21     # window for post-processing median filter
    VARIANCE_SMOOTHING    = 1e-3   # added to all variances for stability
    PREDICTION_CHUNK_SIZE = 500_000  # pixels per prediction batch (mem control)

    # Number of features per pixel (R,G,B,H,S,V,ExG,local_std)
    NUM_FEATURES = 8

    # The five class IDs we model are in fixed order as in the brief.
    CLASSES = np.array([1, 2, 3, 4, 5], dtype=np.int16)

    def __init__(self):
        """
        Initialise the segmenter. The brief mandates no inputs to __init__,
        so we only set up empty containers for the parameters that will be
        learnt during training.
        """
        # Per-class, per-feature means and variances. Shapes (K, F).
        # Populated by model_training().
        self.means     = None
        self.variances = None

        # We deliberately use a UNIFORM prior (1/K for every class) rather
        # than estimating P(class) from training pixel counts. This is because
        # the marking metric is balanced accuracy (mean recall across classes),
        # and the training mask is heavily imbalanced (~52% grass, ~1% vehicle).
        # An empirical prior would push the model toward majority classes and
        # hurt balanced accuracy.
        K = len(self.CLASSES)
        self.log_prior = np.log(np.full(K, 1.0 / K))

    # -------------------------------------------------------------------------
    #  PUBLIC API REQUIRED BY THE TEMPLATE
    # -------------------------------------------------------------------------

    def model_training(self, Training_Image_Name, Training_Image_Mask_Name):
        """
        Train the Naive Bayes model from one image and its label mask.

        For each class k and each feature j we compute the mean mu[k, j]
        and the variance sigma^2[k, j] of feature j across all pixels of
        class k. These sufficient statistics define the Gaussian likelihood
        used at prediction time. Training is closed-form: no iteration,
        no learning rate, no gradient descent.

        Inputs are filenames (per the template); both files must be
        readable from the current working directory.
        """
        # --- Load training image and mask -----------------------------------
        image = cv2.imread(Training_Image_Name)
        if image is None:
            raise FileNotFoundError(
                f"Could not read training image '{Training_Image_Name}'.")
        mask = cv2.imread(Training_Image_Mask_Name)
        if mask is None:
            raise FileNotFoundError(
                f"Could not read training mask '{Training_Image_Mask_Name}'.")
        # The mask has 3 identical channels per the brief; take channel 0.
        mask = mask[:, :, 0]

        # --- Extract per-pixel features --------------------------------------
        features, _, _ = self._extract_features(image)
        labels = mask.flatten()

        # --- Filter out any pixels with invalid labels -----------------------
        # Defensive: keep only pixels whose label is one of the five real
        # classes. This guards against any stray label values in the mask.
        valid = (labels >= 1) & (labels <= 5)
        features = features[valid]
        labels   = labels[valid]

        # --- Fit Gaussian per (class, feature) -------------------------------
        K = len(self.CLASSES)
        F = self.NUM_FEATURES
        self.means     = np.zeros((K, F), dtype=np.float64)
        self.variances = np.zeros((K, F), dtype=np.float64)

        for i, class_id in enumerate(self.CLASSES):
            # Pick out only this class's pixels.
            class_mask = (labels == class_id)
            class_feats = features[class_mask]
            # Compute mean and variance per feature.
            self.means[i]     = class_feats.mean(axis=0)
            # Add a tiny constant to variances to avoid divide-by-zero in
            # the log-Gaussian if any feature has near-zero spread.
            self.variances[i] = class_feats.var(axis=0) + self.VARIANCE_SMOOTHING

    def model_testing(self, Testing_Image_Name, Testing_Image_Mask_Name):
        """
        Predict pixel-wise class labels for a test image and save the result
        as a 3-channel PNG with identical channels (the format the brief
        prescribes for the output mask).
        """
        if self.means is None:
            raise RuntimeError(
                "Model has not been trained. Call model_training() first.")

        # --- Load and feature-extract the test image -------------------------
        image = cv2.imread(Testing_Image_Name)
        if image is None:
            raise FileNotFoundError(
                f"Could not read test image '{Testing_Image_Name}'.")
        features, H, W = self._extract_features(image)

        # --- Predict per-pixel class labels ---------------------------------
        predictions = self._predict(features)
        pred_mask = predictions.reshape(H, W).astype(np.uint8)

        # --- Post-process with a median filter ------------------------------
        # The raw per-pixel predictions are noisy because each pixel is
        # classified independently. A median filter with an appropriately
        # sized window removes salt-and-pepper noise while preserving
        # large class regions and most object boundaries.
        clean_mask = median_filter(
            pred_mask, size=self.MEDIAN_FILTER_SIZE
        ).astype(np.uint8)

        # --- Save in the required format -------------------------------------
        # 3-channel PNG with all channels equal to the class label.
        out_mask = np.stack([clean_mask, clean_mask, clean_mask], axis=-1)
        cv2.imwrite(Testing_Image_Mask_Name, out_mask)

    # -------------------------------------------------------------------------
    #  PRIVATE HELPERS
    # -------------------------------------------------------------------------

    def _extract_features(self, image_bgr):
        """
        Compute an 8-dimensional feature vector for every pixel.

        Features (in order):
            R, G, B            : raw colour
            H, S, V            : hue, saturation, value (HSV colour space)
            ExG = 2*G - R - B  : excess-green vegetation index
            local_std          : standard deviation of V over a small window,
                                 a smoothness/texture descriptor

        Returns
        -------
        features : (H*W, 8) float32, one row per pixel
        H, W     : original image dimensions, returned so the caller can
                   reshape predictions back into a 2-D mask
        """
        H, W = image_bgr.shape[:2]
        img_f = image_bgr.astype(np.float32)

        # --- Colour features in BGR -----------------------------------------
        # OpenCV uses BGR channel order on imread.
        B = img_f[:, :, 0]
        G = img_f[:, :, 1]
        R = img_f[:, :, 2]

        # --- Colour features in HSV -----------------------------------------
        # HSV separates colour (H, S) from brightness (V). Saturation in
        # particular is highly discriminative for grey man-made surfaces
        # (low S) versus saturated vegetation (high S).
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
        H_chan = hsv[:, :, 0]
        S_chan = hsv[:, :, 1]
        V_chan = hsv[:, :, 2]

        # --- Excess-green vegetation index ----------------------------------
        # ExG is large and positive for green pixels, near zero for grey,
        # and negative for red/brown/blue-dominated pixels. Cheap, effective
        # vegetation detector borrowed from the remote-sensing literature.
        ExG = 2.0 * G - R - B

        # --- Local standard deviation of V (texture descriptor) -------------
        # We use the identity Var[X] = E[X^2] - (E[X])^2, computed via two
        # box blurs. This avoids an explicit nested loop over neighbourhoods
        # and is several orders of magnitude faster than the naive approach.
        # The np.maximum ensures checks against tiny negative values.
        ks = (self.STD_WINDOW_SIZE, self.STD_WINDOW_SIZE)
        mean_V    = cv2.blur(V_chan,        ks)
        mean_V_sq = cv2.blur(V_chan ** 2,   ks)
        local_std = np.sqrt(np.maximum(mean_V_sq - mean_V ** 2, 0.0))

        # --- Stack into (H, W, 8) and flatten to (H*W, 8) -------------------
        feature_stack = np.stack(
            [R, G, B, H_chan, S_chan, V_chan, ExG, local_std], axis=-1
        ).astype(np.float32)

        return feature_stack.reshape(-1, self.NUM_FEATURES), H, W

    def _predict(self, features):
        """
        Predict a class label for every pixel using Gaussian Naive Bayes.

        For each pixel n and class k we compute the log-posterior:

            score[n, k] = log P(class=k)
                        + sum over features j of log N( f_nj ; mu_kj, var_kj )

        where the per-feature log-Gaussian is

            log N(f ; mu, var) = -0.5 * log(2*pi*var) - (f - mu)^2 / (2*var)

        Working in log-space turns products of small probabilities into
        sums and avoids floating-point underflow. The naive independence
        assumption (factorising the joint likelihood across features) is
        why this is "Naive" Bayes; it is approximate but allows the
        per-feature 1-D Gaussians used here.

        We process pixels in chunks to keep peak memory bounded - on a
        12-million-pixel image, a single (N, F) intermediate array of
        float64 would require ~770 MB.
        """
        K = len(self.CLASSES)
        N, F = features.shape

        # Pre-compute the constant term per (class, feature). Shape (K, F).
        log_norm_const = -0.5 * np.log(2.0 * np.pi * self.variances)

        predictions = np.empty(N, dtype=np.int16)

        for start in range(0, N, self.PREDICTION_CHUNK_SIZE):
            end = min(start + self.PREDICTION_CHUNK_SIZE, N)
            chunk = features[start:end].astype(np.float64)  # (n, F)

            # Score this chunk against each class. log_post has shape (n, K).
            log_post = np.empty((end - start, K), dtype=np.float64)
            for k in range(K):
                # Squared deviation per (pixel, feature). Broadcasting:
                # (n, F) - (F,) -> (n, F).
                diff_sq = (chunk - self.means[k]) ** 2
                # Per-feature log-likelihood, then sum across features.
                log_gauss = log_norm_const[k] - diff_sq / (2.0 * self.variances[k])
                log_post[:, k] = log_gauss.sum(axis=1) + self.log_prior[k]

            # Pick the class with the largest log-posterior for each pixel.
            best_idx = np.argmax(log_post, axis=1)
            predictions[start:end] = self.CLASSES[best_idx]

        return predictions


# %% Driver code from the template - DO NOT MODIFY
# -----------------------------------------------------------------------------
# This is the exact block from EEEM005AnswerSheetTemplate.py and must run
# unchanged. Data files are assumed to be in the same working directory.
# -----------------------------------------------------------------------------
Model = EEEM005_Coursework_Solution()
Model.model_training("training_image.jpg", "training_mask.png")
Model.model_testing("testing_image1.jpg", "testing_mask1.png")
Model.model_testing("testing_image2.jpg", "testing_mask2.png")