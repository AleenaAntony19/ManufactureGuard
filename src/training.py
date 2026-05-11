"""
training.py — Stage 4: Feature Pipeline & Model Training
=========================================================
MULTICLASS: y = class_idx (0-5). SVM, RF, GB all support multiclass natively.
"""

from __future__ import annotations

import time
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler

from src.config import (
    SVM_MODEL_PATH, RF_MODEL_PATH, GB_MODEL_PATH,
    SCALER_PATH, SELECTOR_PATH, LABEL_MAP_PATH, FEATURE_NAMES_PATH,
    N_FEATURES_TO_SELECT, RANDOM_STATE, NEU_CLASSES_SORTED,
)
from src.feature_extraction import extract_all_features, get_feature_names
from src.models import build_svm, build_random_forest, build_gradient_boosting


def extract_features_batch(paths: List[str], verbose: bool = True) -> np.ndarray:
    features = []
    n = len(paths)
    t0 = time.time()
    for i, path in enumerate(paths):
        try:
            vec = extract_all_features(path)
        except Exception as exc:
            print(f"  [WARN] Skipping {path}: {exc}")
            vec = np.zeros(features[0].shape if features else (108,), dtype=np.float32)
        features.append(vec)
        if verbose and (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            remaining = (n - i - 1) / max(rate, 1e-6)
            print(f"  [{i+1:4d}/{n}] {rate:.1f} img/s  ~{remaining:.0f}s remaining")
    arr = np.vstack(features).astype(np.float32)
    print(f"  Feature matrix: {arr.shape}  ({time.time()-t0:.1f}s total)")
    return arr


def build_preprocessing_pipeline(
    X_train: np.ndarray, y_train: np.ndarray
) -> Tuple[StandardScaler, SelectKBest]:
    print("\n[Pipeline] Fitting StandardScaler …")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    k = min(N_FEATURES_TO_SELECT, X_scaled.shape[1])
    print(f"[Pipeline] SelectKBest top-{k} (f_classif, multiclass) …")
    selector = SelectKBest(score_func=f_classif, k=k)
    selector.fit(X_scaled, y_train)
    print(f"[Pipeline] Selected {selector.get_support().sum()} features from {X_scaled.shape[1]}.")
    return scaler, selector


def apply_preprocessing(X: np.ndarray, scaler, selector) -> np.ndarray:
    return selector.transform(scaler.transform(X))


def train_all_models(train_df: pd.DataFrame, val_df: pd.DataFrame) -> Dict:
    print("\n" + "=" * 60)
    print("  PHASE 1: FEATURE EXTRACTION")
    print("=" * 60)

    print(f"\n[Train] {len(train_df)} images …")
    X_train_raw = extract_features_batch(train_df["image_path"].tolist())
    y_train     = train_df["label"].to_numpy()   # class idx 0-5

    print(f"\n[Val]   {len(val_df)} images …")
    X_val_raw = extract_features_batch(val_df["image_path"].tolist())
    y_val     = val_df["label"].to_numpy()

    # Sanity check — must have all 6 classes in training set
    unique_classes = np.unique(y_train)
    print(f"\n[Train] Unique class labels: {unique_classes}  (expect 0-5)")
    assert len(unique_classes) > 1, \
        f"Only {len(unique_classes)} class(es) in training set — need at least 2!"

    all_feature_names = get_feature_names()

    print("\n" + "=" * 60)
    print("  PHASE 2: PREPROCESSING")
    print("=" * 60)

    scaler, selector = build_preprocessing_pipeline(X_train_raw, y_train)
    X_train = apply_preprocessing(X_train_raw, scaler, selector)
    X_val   = apply_preprocessing(X_val_raw,   scaler, selector)

    support = selector.get_support()
    selected_feature_names = [n for n, s in zip(all_feature_names, support) if s]

    # label_map: class_idx → class_name
    label_map = {int(row["label"]): row["class_name"]
                 for _, row in train_df.drop_duplicates("label").iterrows()}

    print("\n" + "=" * 60)
    print("  PHASE 3: MODEL TRAINING  (MULTICLASS — 6 defect types)")
    print("=" * 60)

    # ── SVM ──────────────────────────────────────────────────────
    print("\n[SVM] Training …")
    svm = build_svm()
    t0 = time.time()
    svm.fit(X_train, y_train)
    print(f"  Done in {time.time()-t0:.1f}s")
    val_acc = (svm.predict(X_val) == y_val).mean()
    print(f"  Val accuracy: {val_acc:.4f}")

    # ── Random Forest ──────────────────────────────────────────
    print("\n[RF]  Training …")
    rf = build_random_forest()
    t0 = time.time()
    rf.fit(X_train, y_train)
    print(f"  Done in {time.time()-t0:.1f}s")
    val_acc = (rf.predict(X_val) == y_val).mean()
    print(f"  Val accuracy: {val_acc:.4f}")

    # ── Gradient Boosting ───────────────────────────────────────
    print("\n[GB]  Training …")
    gb = build_gradient_boosting()
    t0 = time.time()
    gb.fit(X_train, y_train)
    print(f"  Done in {time.time()-t0:.1f}s")
    val_acc = (gb.predict(X_val) == y_val).mean()
    print(f"  Val accuracy: {val_acc:.4f}")

    print("\n" + "=" * 60)
    print("  PHASE 4: SAVING ARTEFACTS")
    print("=" * 60)
    joblib.dump(svm,                   SVM_MODEL_PATH);    print(f"  Saved → {SVM_MODEL_PATH}")
    joblib.dump(rf,                    RF_MODEL_PATH);     print(f"  Saved → {RF_MODEL_PATH}")
    joblib.dump(gb,                    GB_MODEL_PATH);     print(f"  Saved → {GB_MODEL_PATH}")
    joblib.dump(scaler,                SCALER_PATH);       print(f"  Saved → {SCALER_PATH}")
    joblib.dump(selector,              SELECTOR_PATH);     print(f"  Saved → {SELECTOR_PATH}")
    joblib.dump(label_map,             LABEL_MAP_PATH);    print(f"  Saved → {LABEL_MAP_PATH}")
    joblib.dump(selected_feature_names, FEATURE_NAMES_PATH)
    print(f"  Saved → {FEATURE_NAMES_PATH}")

    return dict(svm=svm, rf=rf, gb=gb, scaler=scaler, selector=selector,
                label_map=label_map, feature_names=selected_feature_names)
