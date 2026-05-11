"""
data_loader.py — Stage 1: Dataset Loading & Scanning
======================================================
MULTICLASS MODE: label = class_idx (0-5), one per defect type.

Handles the actual NEU-DET zip layout:
  data/NEU-DET/IMAGES/crazing_1.jpg
  data/NEU-DET/IMAGES/inclusion_5.jpg  ...

Class is parsed from the filename prefix.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    DATA_DIR, RANDOM_STATE,
    TRAIN_RATIO, VAL_RATIO, TEST_RATIO,
    NEU_CLASSES, NEU_CLASSES_SORTED,
    BINARY_DEFECTIVE_LABEL,
)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

_CLASS_PREFIXES = {
    "crazing":         ["crazing"],
    "inclusion":       ["inclusion"],
    "patches":         ["patches"],
    "pitted_surface":  ["pitted_surface", "pitted"],
    "rolled-in_scale": ["rolled-in_scale", "rolled_in_scale", "rs"],
    "scratches":       ["scratches"],
}

# class_name → integer label (0..5, alphabetical)
CLASS_TO_IDX = {cls: i for i, cls in enumerate(NEU_CLASSES_SORTED)}
IDX_TO_CLASS = {i: cls for cls, i in CLASS_TO_IDX.items()}


def _class_from_filename(fname: str) -> Optional[str]:
    stem = Path(fname).stem.lower()
    for canonical, prefixes in _CLASS_PREFIXES.items():
        for pfx in prefixes:
            if stem.startswith(pfx):
                return canonical
    return None


def _collect_flat(images_dir: Path) -> pd.DataFrame:
    rows: List[Dict] = []
    for fpath in sorted(images_dir.iterdir()):
        if fpath.suffix.lower() not in IMG_EXTS:
            continue
        cls = _class_from_filename(fpath.name)
        if cls is None:
            continue
        rows.append({
            "image_path": str(fpath),
            "class_name": cls,
            "label":      CLASS_TO_IDX[cls],     # ← multiclass label 0-5
        })
    return pd.DataFrame(rows)


def _collect_subfolders(base: Path) -> pd.DataFrame:
    rows: List[Dict] = []
    for d in sorted(base.iterdir()):
        if not d.is_dir():
            continue
        cls = next((c for c in NEU_CLASSES if d.name.lower() == c.lower()), None)
        if cls is None:
            continue
        for root_s, _, files in os.walk(d):
            for fname in sorted(files):
                if Path(fname).suffix.lower() in IMG_EXTS:
                    rows.append({
                        "image_path": str(Path(root_s) / fname),
                        "class_name": cls,
                        "label":      CLASS_TO_IDX[cls],
                    })
    return pd.DataFrame(rows)


def _scan_root(root: Path) -> pd.DataFrame:
    # Layout A — flat IMAGES folder (your actual zip)
    for flat_dir in [root/"NEU-DET"/"IMAGES", root/"NEU_DET"/"IMAGES",
                     root/"IMAGES", root/"images"]:
        if flat_dir.exists():
            df = _collect_flat(flat_dir)
            if len(df) > 0:
                print(f"[DataLoader] Layout A (flat filenames): {flat_dir}")
                return df
    # Layout B — class sub-folders at root
    df = _collect_subfolders(root)
    if len(df) > 0:
        print(f"[DataLoader] Layout B (class sub-folders): {root}")
        return df
    # Layout C — nested
    for sub in ["IMAGES", "images", "NEU-DET", "NEU_DET"]:
        s = root / sub
        if s.exists():
            df = _collect_subfolders(s)
            if len(df) > 0:
                print(f"[DataLoader] Layout C (nested): {s}")
                return df
    return pd.DataFrame()


def load_dataset(data_dir=None) -> pd.DataFrame:
    root = Path(data_dir or DATA_DIR)
    if not root.exists():
        raise FileNotFoundError(
            f"Data directory not found: {root}\n"
            "Unzip archive.zip into the data/ folder so that\n"
            "data/NEU-DET/IMAGES/crazing_1.jpg exists."
        )
    df = _scan_root(root)
    if len(df) == 0:
        raise FileNotFoundError(
            f"No NEU images found under: {root}\n"
            "Make sure data/NEU-DET/IMAGES/*.jpg exists after unzipping."
        )

    # Also keep class_idx column for backward compat
    df["class_idx"] = df["label"]

    print(f"\n[DataLoader] Found {len(df)} images, {df['class_name'].nunique()} classes "
          f"(MULTICLASS mode — label=class index 0-5):")
    for cls in NEU_CLASSES_SORTED:
        cnt = (df["class_name"] == cls).sum()
        idx = CLASS_TO_IDX[cls]
        print(f"   [{idx}] {cls:25s} → {cnt:4d} images")
    return df.reset_index(drop=True)


def split_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df, temp_df = train_test_split(
        df, test_size=(VAL_RATIO + TEST_RATIO),
        stratify=df["class_name"], random_state=RANDOM_STATE,
    )
    rel_test = TEST_RATIO / (VAL_RATIO + TEST_RATIO)
    val_df, test_df = train_test_split(
        temp_df, test_size=rel_test,
        stratify=temp_df["class_name"], random_state=RANDOM_STATE,
    )
    print(f"\n[DataLoader] Split → train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    return (train_df.reset_index(drop=True),
            val_df.reset_index(drop=True),
            test_df.reset_index(drop=True))
