"""
evaluation.py — Stage 5: Multiclass Evaluation
================================================
Metrics for 6-class defect classification:
  - Accuracy, macro F1, per-class precision/recall
  - Confusion matrix (6×6)
  - ROC-AUC (one-vs-rest)
  - Feature importance per model
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.preprocessing import label_binarize

from src.config import (
    OUTPUT_DIR, COLORS, CLASS_COLORS,
    FIGSIZE_LARGE, FIGSIZE_MEDIUM, DPI,
    NEU_CLASSES_SORTED, N_CLASSES,
)


def compute_metrics(y_true, y_pred, y_prob, model_name="") -> Dict:
    """Multiclass metrics — macro-averaged."""
    metrics = {
        "model":          model_name,
        "accuracy":       accuracy_score(y_true, y_pred),
        "macro_f1":       f1_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall":   recall_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1":    f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }
    # ROC-AUC one-vs-rest (needs probability matrix)
    try:
        y_bin = label_binarize(y_true, classes=list(range(N_CLASSES)))
        metrics["roc_auc_ovr"] = roc_auc_score(y_bin, y_prob, multi_class="ovr",
                                                average="macro")
    except Exception:
        metrics["roc_auc_ovr"] = float("nan")
    return metrics


def plot_confusion_matrix(y_true, y_pred, model_name="", save_path=None):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(N_CLASSES)))
    labels = [c[:10] for c in NEU_CLASSES_SORTED]   # shorten for display

    fig, ax = plt.subplots(figsize=(8, 7), dpi=DPI)
    fig.patch.set_facecolor(COLORS["bg_dark"])
    ax.set_facecolor(COLORS["bg_card"])

    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)

    ax.set_xticks(range(N_CLASSES)); ax.set_yticks(range(N_CLASSES))
    ax.set_xticklabels(labels, rotation=35, ha="right",
                       color=COLORS["text"], fontsize=9)
    ax.set_yticklabels(labels, color=COLORS["text"], fontsize=9)
    ax.set_xlabel("Predicted", color=COLORS["text"], fontsize=12)
    ax.set_ylabel("True",      color=COLORS["text"], fontsize=12)
    ax.set_title(f"Confusion Matrix — {model_name}",
                 color=COLORS["text"], fontsize=13)

    thresh = cm.max() / 2.0
    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=10, fontweight="bold")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_roc_ovr(y_true, y_prob_matrix, model_name="", save_path=None):
    """One-vs-rest ROC curves for all 6 classes."""
    y_bin = label_binarize(y_true, classes=list(range(N_CLASSES)))
    from sklearn.metrics import roc_curve, auc

    fig, ax = plt.subplots(figsize=FIGSIZE_MEDIUM, dpi=DPI)
    fig.patch.set_facecolor(COLORS["bg_dark"])
    ax.set_facecolor(COLORS["bg_card"])

    for i, cls in enumerate(NEU_CLASSES_SORTED):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob_matrix[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=CLASS_COLORS[i], lw=2,
                label=f"{cls} (AUC={roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "w--", alpha=0.4, lw=1.5)
    ax.set_xlabel("False Positive Rate", color=COLORS["text"], fontsize=12)
    ax.set_ylabel("True Positive Rate",  color=COLORS["text"], fontsize=12)
    ax.set_title(f"ROC Curves (One-vs-Rest) — {model_name}",
                 color=COLORS["text"], fontsize=14)
    ax.legend(facecolor=COLORS["bg_card"], labelcolor=COLORS["text"],
              fontsize=9, framealpha=0.85)
    ax.tick_params(colors=COLORS["text"])
    for sp in ax.spines.values(): sp.set_edgecolor(COLORS["grid"])

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_feature_importance(model, feature_names, model_name="",
                            top_n=25, save_path=None):
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_).mean(axis=0)
    else:
        return None

    n = min(top_n, len(importances))
    idx = np.argsort(importances)[-n:][::-1]
    names = [feature_names[i] if i < len(feature_names) else f"f{i}" for i in idx]
    vals  = importances[idx]

    fig, ax = plt.subplots(figsize=(10, max(6, n * 0.32)), dpi=DPI)
    fig.patch.set_facecolor(COLORS["bg_dark"])
    ax.set_facecolor(COLORS["bg_card"])
    ax.barh(range(n), vals[::-1], color=COLORS["primary"], alpha=0.85)
    ax.set_yticks(range(n))
    ax.set_yticklabels(names[::-1], color=COLORS["text"], fontsize=8)
    ax.set_xlabel("Importance", color=COLORS["text"])
    ax.set_title(f"Top-{n} Features — {model_name}", color=COLORS["text"], fontsize=13)
    ax.tick_params(colors=COLORS["text"])
    for sp in ax.spines.values(): sp.set_edgecolor(COLORS["grid"])

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def evaluate_all_models(models_dict, X_test, y_test,
                        class_names_test=None, feature_names=None) -> pd.DataFrame:
    all_metrics = []

    for key, label in [("svm","SVM"), ("rf","Random Forest"), ("gb","Gradient Boosting")]:
        if key not in models_dict:
            continue
        model  = models_dict[key]
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)   # shape (N, 6)

        metrics = compute_metrics(y_test, y_pred, y_prob, model_name=label)
        all_metrics.append(metrics)

        print(f"\n{'─'*50}\n  {label}\n{'─'*50}")
        for k, v in metrics.items():
            if k != "model":
                print(f"  {k:22s}: {v:.4f}")
        print("\n  Per-class report:")
        print(classification_report(y_test, y_pred,
                                    target_names=NEU_CLASSES_SORTED,
                                    zero_division=0))

        plot_confusion_matrix(y_test, y_pred, model_name=label,
                              save_path=OUTPUT_DIR / f"cm_{key}.png")
        plt.close("all")

        plot_roc_ovr(y_test, y_prob, model_name=label,
                     save_path=OUTPUT_DIR / f"roc_{key}.png")
        plt.close("all")

        if feature_names:
            plot_feature_importance(model, feature_names, model_name=label,
                                    save_path=OUTPUT_DIR / f"importance_{key}.png")
            plt.close("all")

    df = pd.DataFrame(all_metrics).set_index("model")
    print("\n" + "=" * 60)
    print("  EVALUATION SUMMARY")
    print("=" * 60)
    print(df.to_string())
    return df
