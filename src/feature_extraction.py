"""
feature_extraction.py — Stage 2: Texture Feature Extraction
=============================================================
Extracts three families of texture features from grayscale images:

1. LBP  — Local Binary Patterns (multi-scale)
2. GLCM — Gray-Level Co-occurrence Matrix statistics
           (energy, contrast, dissimilarity, homogeneity, ASM, correlation)
3. Gabor — Filter bank responses at multiple orientations × frequencies

Each image yields a single 1-D feature vector that concatenates all three.
"""

from __future__ import annotations

import warnings
from typing import List, Tuple

try:
    import cv2
except ImportError:
    import subprocess,sys; subprocess.check_call([sys.executable,'-m','pip','install','opencv-python-headless']); import cv2
import numpy as np
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops

from src.config import (
    IMG_SIZE,
    LBP_RADIUS, LBP_N_POINTS, LBP_METHOD,
    GLCM_DISTANCES, GLCM_ANGLES, GLCM_PROPERTIES,
    GABOR_FREQUENCIES, GABOR_ORIENTATIONS,
)


# ══════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════

def load_grayscale(path: str) -> np.ndarray:
    """Load an image from *path* as a uint8 grayscale array of shape IMG_SIZE."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    img = cv2.resize(img, IMG_SIZE, interpolation=cv2.INTER_AREA)
    return img


def _safe_normalize(arr: np.ndarray) -> np.ndarray:
    """Normalize to [0, 1]; return zeros if all values identical."""
    rng = arr.max() - arr.min()
    if rng < 1e-12:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - arr.min()) / rng).astype(np.float32)


# ══════════════════════════════════════════════════════════
# 1. LBP Features
# ══════════════════════════════════════════════════════════

def extract_lbp(gray: np.ndarray) -> np.ndarray:
    """
    Extract multi-scale LBP histogram features.

    For each (radius, n_points) pair we compute a uniform LBP image,
    then form a normalized histogram of pattern codes.

    Returns
    -------
    np.ndarray, shape (sum of (n_points+2) for each scale,)
    """
    feats: List[np.ndarray] = []
    for radius, n_points in zip(LBP_RADIUS, LBP_N_POINTS):
        lbp = local_binary_pattern(gray, n_points, radius, method=LBP_METHOD)
        n_bins = n_points + 2           # uniform has n_points+2 bins
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins,
                               range=(0, n_bins), density=True)
        feats.append(hist.astype(np.float32))
    return np.concatenate(feats)


# ══════════════════════════════════════════════════════════
# 2. GLCM Features
# ══════════════════════════════════════════════════════════

def extract_glcm(gray: np.ndarray) -> np.ndarray:
    """
    Extract GLCM-based texture statistics.

    Computes co-occurrence matrices for all (distance, angle) combinations,
    then averages each property over angles for each distance.

    Properties: energy, contrast, dissimilarity, homogeneity, ASM, correlation

    Returns
    -------
    np.ndarray, shape (n_distances × n_properties,)
    """
    # GLCM requires uint8 input with levels=256
    feats: List[float] = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        glcm = graycomatrix(
            gray,
            distances=GLCM_DISTANCES,
            angles=GLCM_ANGLES,
            levels=256,
            symmetric=True,
            normed=True,
        )
        # glcm shape: (levels, levels, n_distances, n_angles)
        for prop in GLCM_PROPERTIES:
            vals = graycoprops(glcm, prop)   # shape (n_distances, n_angles)
            # Average over angles → one value per distance
            feats.extend(vals.mean(axis=1).tolist())

    return np.array(feats, dtype=np.float32)


# ══════════════════════════════════════════════════════════
# 3. Gabor Filter Features
# ══════════════════════════════════════════════════════════

def _build_gabor_kernel(frequency: float, theta: float) -> np.ndarray:
    """Build a Gabor kernel via cv2 for a given frequency and orientation."""
    wavelength = 1.0 / (frequency + 1e-12)
    sigma      = wavelength * 0.56          # rule-of-thumb bandwidth ~1 octave
    gamma      = 0.5                        # spatial aspect ratio
    ksize      = int(6 * sigma + 1) | 1    # must be odd
    ksize      = max(ksize, 3)
    kernel     = cv2.getGaborKernel(
        (ksize, ksize), sigma, theta,
        wavelength, gamma, psi=0,
        ktype=cv2.CV_32F
    )
    return kernel


def extract_gabor(gray: np.ndarray) -> np.ndarray:
    """
    Extract Gabor filter-bank statistics.

    For each (frequency, orientation) pair we convolve the image with
    a Gabor kernel and collect mean + std of the absolute response.

    Returns
    -------
    np.ndarray, shape (n_frequencies × n_orientations × 2,)
    """
    img_f32 = gray.astype(np.float32) / 255.0
    feats: List[float] = []

    for freq in GABOR_FREQUENCIES:
        for theta in GABOR_ORIENTATIONS:
            kernel   = _build_gabor_kernel(freq, theta)
            filtered = cv2.filter2D(img_f32, cv2.CV_32F, kernel)
            response = np.abs(filtered)
            feats.append(float(response.mean()))
            feats.append(float(response.std()))

    return np.array(feats, dtype=np.float32)


# ══════════════════════════════════════════════════════════
# 4. Combined Feature Vector
# ══════════════════════════════════════════════════════════

def extract_all_features(path: str) -> np.ndarray:
    """
    Load an image and return the concatenated [LBP | GLCM | Gabor] feature vector.

    Parameters
    ----------
    path : str
        Filesystem path to the image file.

    Returns
    -------
    np.ndarray, shape (D,) — float32
    """
    gray = load_grayscale(path)
    lbp_feats   = extract_lbp(gray)
    glcm_feats  = extract_glcm(gray)
    gabor_feats = extract_gabor(gray)
    return np.concatenate([lbp_feats, glcm_feats, gabor_feats])


def extract_features_from_array(img_array: np.ndarray) -> np.ndarray:
    """
    Extract features from a numpy image array (already loaded, BGR or gray).

    Accepts BGR (3-channel) or single-channel uint8 arrays.
    Resizes to IMG_SIZE internally.
    """
    if img_array.ndim == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_array.copy()

    gray = cv2.resize(gray, IMG_SIZE, interpolation=cv2.INTER_AREA)
    if gray.dtype != np.uint8:
        gray = np.clip(gray, 0, 255).astype(np.uint8)

    lbp_feats   = extract_lbp(gray)
    glcm_feats  = extract_glcm(gray)
    gabor_feats = extract_gabor(gray)
    return np.concatenate([lbp_feats, glcm_feats, gabor_feats])


def get_feature_names() -> List[str]:
    """Return human-readable names for every position in the feature vector."""
    names: List[str] = []

    # LBP
    for r, n in zip(LBP_RADIUS, LBP_N_POINTS):
        for b in range(n + 2):
            names.append(f"lbp_r{r}_bin{b}")

    # GLCM
    for prop in GLCM_PROPERTIES:
        for d in GLCM_DISTANCES:
            names.append(f"glcm_{prop}_d{d}")

    # Gabor
    for fi, freq in enumerate(GABOR_FREQUENCIES):
        for oi, theta in enumerate(GABOR_ORIENTATIONS):
            names.append(f"gabor_f{fi}_t{oi}_mean")
            names.append(f"gabor_f{fi}_t{oi}_std")

    return names
