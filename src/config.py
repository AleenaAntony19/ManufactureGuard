"""
config.py — Central Configuration
===================================
Single source of truth for all paths, hyperparameters, feature
definitions, and constants used across every module in the project.

Environment Variables:
  NEU_DATA_DIR: Path to NEU Surface Defect dataset folder
                Defaults to ./data if not set.

CLASSIFICATION MODE: MULTICLASS (6 defect types)
  The NEU-DET dataset has NO non-defective images.
  All 1800 images are defective. We classify WHICH of 6 defect
  types each image belongs to (label = class index 0-5).
"""

import os
from pathlib import Path

# ─────────────────────────────── Paths ───────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent

DATA_DIR   = Path(os.getenv("NEU_DATA_DIR", PROJECT_ROOT / "data"))
MODEL_DIR  = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

for _d in [MODEL_DIR, OUTPUT_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ───────────────────── Model Artifact Paths ─────────────────────
SVM_MODEL_PATH     = MODEL_DIR / "svm_model.pkl"
RF_MODEL_PATH      = MODEL_DIR / "rf_model.pkl"
GB_MODEL_PATH      = MODEL_DIR / "gb_model.pkl"
SCALER_PATH        = MODEL_DIR / "scaler.pkl"
SELECTOR_PATH      = MODEL_DIR / "selector.pkl"
LABEL_MAP_PATH     = MODEL_DIR / "label_map.pkl"
FEATURE_NAMES_PATH = MODEL_DIR / "feature_names.pkl"

# ───────────────────── Dataset Constants ─────────────────────
RANDOM_STATE = 42
IMG_SIZE     = (200, 200)
TRAIN_RATIO  = 0.70
VAL_RATIO    = 0.15
TEST_RATIO   = 0.15

# NEU-DET: 6 defect classes, 300 images each = 1800 total
# ALL images are defective — we do MULTICLASS (which defect type)
NEU_CLASSES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]
NEU_CLASSES_SORTED = sorted(NEU_CLASSES)   # alphabetical, determines label index
N_CLASSES = len(NEU_CLASSES)               # 6

# label = class_idx (0..5), one per defect type
# kept for legacy compatibility only:
BINARY_DEFECTIVE_LABEL = 1
BINARY_OK_LABEL        = 0

# ───────────────────── Feature Extraction ─────────────────────
LBP_RADIUS   = [1, 2, 3]
LBP_N_POINTS = [8, 16, 24]
LBP_METHOD   = "uniform"

GLCM_DISTANCES  = [1, 3, 5]
GLCM_ANGLES     = [0, 0.785, 1.571, 2.356]
GLCM_PROPERTIES = ["energy", "contrast", "dissimilarity",
                   "homogeneity", "ASM", "correlation"]

GABOR_FREQUENCIES  = [0.1, 0.25, 0.4]
GABOR_ORIENTATIONS = [0, 0.393, 0.785, 1.178, 1.571, 1.963]

N_FEATURES_TO_SELECT = 80

# ───────────────────── Model Hyperparameters ─────────────────────

# SVM — multiclass via one-vs-one (sklearn default)
SVM_PARAMS = {
    "C":            10.0,
    "kernel":       "rbf",
    "gamma":        "scale",
    "probability":  True,
    "class_weight": "balanced",
    "random_state": RANDOM_STATE,
    "max_iter":     5000,
    "decision_function_shape": "ovr",
}

# Random Forest
RF_PARAMS = {
    "n_estimators":      400,
    "max_depth":         None,
    "min_samples_split": 4,
    "min_samples_leaf":  2,
    "class_weight":      "balanced",
    "n_jobs":            -1,
    "random_state":      RANDOM_STATE,
}

# Gradient Boosting
GB_PARAMS = {
    "n_estimators":      300,
    "max_depth":         5,
    "learning_rate":     0.08,
    "subsample":         0.85,
    "min_samples_split": 10,
    "random_state":      RANDOM_STATE,
}

DEFAULT_THRESHOLD = 0.50

# ───────────────────── Visualization ─────────────────────
COLORS = {
    "defective":  "#FF4B4B",
    "ok":         "#00D26A",
    "primary":    "#6C63FF",
    "secondary":  "#FF6B6B",
    "accent":     "#4ECDC4",
    "bg_dark":    "#0E1117",
    "bg_card":    "#1A1F2E",
    "text":       "#FAFAFA",
    "grid":       "#2D3748",
}

CLASS_COLORS = [
    "#6C63FF", "#FF6B6B", "#4ECDC4",
    "#FFD93D", "#95E1D3", "#F38181",
]

FIGSIZE_LARGE  = (16, 10)
FIGSIZE_MEDIUM = (12, 7)
FIGSIZE_SMALL  = (8, 5)
DPI = 150
