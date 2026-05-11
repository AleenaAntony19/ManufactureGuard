#!/usr/bin/env python3
"""
train.py — Main Training Entry Point
=====================================
Run this script to train all models on the NEU Surface Defect dataset.

Usage:
    python train.py
    NEU_DATA_DIR=/path/to/neu python train.py

The script:
  1. Scans the dataset directory
  2. Extracts LBP + GLCM + Gabor texture features
  3. Scales and applies feature selection
  4. Trains SVM, Random Forest, and Gradient Boosting
  5. Evaluates on the held-out test set
  6. Saves all artefacts to ./models/
  7. Saves evaluation plots to ./outputs/
"""

from __future__ import annotations

import sys
import time

import joblib
import numpy as np

from src.config import (
    SCALER_PATH, SELECTOR_PATH, FEATURE_NAMES_PATH,
    SVM_MODEL_PATH, RF_MODEL_PATH, GB_MODEL_PATH,
)
from src.data_loader import load_dataset, split_dataset
from src.training import (
    train_all_models,
    extract_features_batch,
    apply_preprocessing,
)
from src.evaluation import evaluate_all_models


def main():
    t_start = time.time()
    print("\n" + "═" * 60)
    print("  ManufactureGuard — Training Pipeline")
    print("  NEU Surface Defect Dataset")
    print("═" * 60)

    # ── 1. Load & split ───────────────────────────────────
    try:
        df = load_dataset()
    except FileNotFoundError as exc:
        print(f"\n[ERROR] {exc}")
        print("\nTo fix:\n"
              "  1. Download NEU-DET from Kaggle:\n"
              "     kaggle datasets download kaustubhdikshit/neu-surface-defect-database\n"
              "  2. Unzip into ./data/\n"
              "  3. Re-run: python train.py\n")
        sys.exit(1)

    train_df, val_df, test_df = split_dataset(df)

    # ── 2. Train (includes feature extraction + saving) ───
    artefacts = train_all_models(train_df, val_df)

    # ── 3. Evaluate on test set ───────────────────────────
    print("\n" + "═" * 60)
    print("  TEST SET EVALUATION")
    print("═" * 60)

    scaler   = artefacts["scaler"]
    selector = artefacts["selector"]
    feat_names = artefacts["feature_names"]

    print(f"\n[Test] Extracting features for {len(test_df)} test images …")
    X_test_raw = extract_features_batch(test_df["image_path"].tolist())
    y_test     = test_df["label"].to_numpy()
    X_test     = apply_preprocessing(X_test_raw, scaler, selector)

    metrics_df = evaluate_all_models(
        models_dict=artefacts,
        X_test=X_test,
        y_test=y_test,
        class_names_test=test_df["class_name"],
        feature_names=feat_names,
    )

    # ── 4. Done ───────────────────────────────────────────
    elapsed = time.time() - t_start
    print(f"\n[Done] Total time: {elapsed/60:.1f} min")
    print("  Models saved to: ./models/")
    print("  Plots  saved to: ./outputs/")
    print("\n  Run the app:\n    streamlit run app.py\n")

    return metrics_df


if __name__ == "__main__":
    main()
