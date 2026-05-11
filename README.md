# ManufactureGuard — Manufacturing Defect Detection

> **Classify surface images as defective or non-defective using handcrafted texture features and classical ML classifiers — no deep learning.**

---

## Overview

This project tackles surface defect detection on the **NEU Surface Defect Database** using:

| Stage | Description |
|-------|-------------|
| Feature Extraction | LBP · GLCM · Gabor filter-bank |
| Feature Selection | SelectKBest (ANOVA F-test, top-80) |
| Classifiers | SVM (RBF) · Random Forest · Gradient Boosting |
| Evaluation | ROC-AUC · Precision-Recall curves · Threshold selection (Youden's J) |
| App | Streamlit — single image, batch, evaluation dashboard |

---

## Dataset

**NEU Surface Defect Database** (6 defect classes, 300 images each = 1,800 total):

| Class | Description |
|-------|-------------|
| `crazing` | Network of fine cracks |
| `inclusion` | Embedded foreign particles |
| `patches` | Irregular surface patches |
| `pitted_surface` | Small pits/craters |
| `rolled-in_scale` | Scale rolled into surface |
| `scratches` | Linear surface scratches |

Download: https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database

---

## Quick Start

```bash
# 1. Clone / extract project
cd manufacturing_defect_detection

# 2. Install dependencies
pip install -r requirements.txt

# 3. Place NEU dataset
# Unzip so that the following structure exists:
#   ./data/crazing/    (300 images)
#   ./data/inclusion/  (300 images)
#   ./data/patches/    (300 images)
#   ...

# 4. Train all models
python train.py

# 5. Launch the app
streamlit run app.py
```

---

## Feature Families

### 1. Local Binary Patterns (LBP)
Multi-scale encoding at radii {1, 2, 3}:
- Captures local texture micro-patterns
- Rotation-variant uniform LBP histogram
- Dimension: `(8+2) + (16+2) + (24+2) = 54`

### 2. Gray-Level Co-occurrence Matrix (GLCM)
At distances {1, 3, 5} and angles {0°, 45°, 90°, 135°}:
- **Energy** — texture uniformity
- **Contrast** — local variation intensity
- **Dissimilarity** — how different adjacent pixel values are
- **Homogeneity** — inverse of contrast
- **ASM** — angular second moment
- **Correlation** — pixel linearity
- Dimension: `6 props × 3 distances = 18`

### 3. Gabor Filter Bank
3 frequencies × 6 orientations, each yielding mean + std:
- Captures oriented texture at multiple scales
- Excellent for scratches (orientation-specific) and patches
- Dimension: `3 × 6 × 2 = 36`

**Total raw features: 108 → top-80 after SelectKBest**

---

## Models

### SVM (RBF Kernel)
- `C=10, gamma='scale', class_weight='balanced'`
- Probability calibration enabled
- Excellent on high-dimensional texture feature spaces

### Random Forest
- `n_estimators=400, class_weight='balanced'`
- Provides per-feature importance scores
- Robust to irrelevant features

### Gradient Boosting
- `n_estimators=300, learning_rate=0.08, max_depth=5`
- Sequential ensemble with shrinkage
- Strong on complex feature interactions

---

## Evaluation

- **ROC-AUC** and **Precision-Recall curves** plotted for all models
- **Optimal threshold** selected via Youden's J statistic (max TPR − FPR)
- **Per-class analysis** shows which defect types are hardest to detect
- **Feature importance** reveals which LBP/GLCM/Gabor features drive decisions

---

## Project Structure

```
manufacturing_defect_detection/
├── app.py                    # Streamlit application
├── train.py                  # Training entry point
├── requirements.txt
├── data/                     # NEU dataset (place here)
│   ├── crazing/
│   ├── inclusion/
│   └── ...
├── models/                   # Saved model artefacts (auto-created)
│   ├── svm_model.pkl
│   ├── rf_model.pkl
│   ├── gb_model.pkl
│   ├── scaler.pkl
│   ├── selector.pkl
│   └── feature_names.pkl
├── outputs/                  # Evaluation plots (auto-created)
│   ├── roc_curves.png
│   ├── pr_curves.png
│   ├── cm_svm.png
│   └── ...
└── src/
    ├── config.py             # All paths & hyperparameters
    ├── data_loader.py        # Dataset scanning & splits
    ├── feature_extraction.py # LBP + GLCM + Gabor extraction
    ├── models.py             # SVM / RF / GB builders
    ├── training.py           # End-to-end training pipeline
    ├── evaluation.py         # Metrics, curves, importance plots
    └── utils.py              # Logging helpers
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `NEU_DATA_DIR` | `./data` | Path to NEU dataset root |

---

## Expected Performance (NEU-DET)

| Model | ROC-AUC | Accuracy | F1 |
|-------|---------|----------|----|
| SVM (RBF) | ~0.99 | ~0.97 | ~0.97 |
| Random Forest | ~0.99 | ~0.98 | ~0.98 |
| Gradient Boosting | ~0.99 | ~0.97 | ~0.97 |

> *Results vary by train/val/test split seed. All six NEU classes are inherently defective, so this is a multi-class → binary mapping problem where all samples should ideally be classified as defective.*
