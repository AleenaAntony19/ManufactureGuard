#!/usr/bin/env python3
"""
ManufactureGuard — Surface Defect Detection App (MULTICLASS)
=============================================================
Classifies images into one of 6 NEU defect types:
  crazing · inclusion · patches · pitted_surface · rolled-in_scale · scratches

Run: streamlit run app.py
"""

from __future__ import annotations

import io, os, time
from pathlib import Path
from typing import Dict, List, Optional

os.environ.setdefault("STREAMLIT_SERVER_FILE_WATCHER_TYPE", "none")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

try:
    import matplotlib
    matplotlib.use("Agg")
except ImportError as exc:
    raise ImportError(
        "Required dependency 'matplotlib' is missing. "
        "Install dependencies with 'pip install -r requirements.txt'."
    ) from exc

try:
    import cv2
except ImportError as exc:
    raise ImportError(
        "Required dependency 'opencv-python-headless' is missing. "
        "Install dependencies with 'pip install -r requirements.txt'."
    ) from exc

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.preprocessing import label_binarize

from src.config import (
    SVM_MODEL_PATH, RF_MODEL_PATH, GB_MODEL_PATH,
    SCALER_PATH, SELECTOR_PATH, FEATURE_NAMES_PATH, LABEL_MAP_PATH,
    COLORS, CLASS_COLORS, NEU_CLASSES_SORTED, N_CLASSES, IMG_SIZE,
)
from src.feature_extraction import extract_features_from_array
from src.data_loader import CLASS_TO_IDX, IDX_TO_CLASS

# ── Page config ─────────────────────────────────────────────────
st.set_page_config(page_title="ManufactureGuard", page_icon="🔬",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
.block-container{padding-top:1.2rem;padding-bottom:2rem;}
.metric-card{border:1px solid rgba(255,255,255,.12);border-radius:14px;
  padding:1rem 1.2rem;background:linear-gradient(180deg,rgba(30,35,50,.92),rgba(16,20,30,.97));
  box-shadow:0 10px 20px rgba(0,0,0,.2);text-align:center;}
.cls-badge{border-radius:8px;padding:5px 14px;font-weight:bold;font-size:1.05rem;display:inline-block;}
.small-note{color:#9aa4b2;font-size:.90rem;}
</style>""", unsafe_allow_html=True)

CLASS_BADGE_COLORS = {
    "crazing":         ("#6C63FF","#1a1535"),
    "inclusion":       ("#FF6B6B","#350f0f"),
    "patches":         ("#4ECDC4","#0d3330"),
    "pitted_surface":  ("#FFD93D","#332e08"),
    "rolled-in_scale": ("#95E1D3","#102a27"),
    "scratches":       ("#F38181","#331313"),
}

# ── Artefact loading ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_artifacts() -> Dict:
    missing = [n for p, n in [
        (SVM_MODEL_PATH,"svm_model.pkl"),(RF_MODEL_PATH,"rf_model.pkl"),
        (GB_MODEL_PATH,"gb_model.pkl"),(SCALER_PATH,"scaler.pkl"),
        (SELECTOR_PATH,"selector.pkl"),(FEATURE_NAMES_PATH,"feature_names.pkl"),
        (LABEL_MAP_PATH,"label_map.pkl"),
    ] if not Path(p).exists()]
    if missing:
        return {"error": f"Missing: {', '.join(missing)}. Run `python train.py` first."}
    return dict(
        svm=joblib.load(SVM_MODEL_PATH), rf=joblib.load(RF_MODEL_PATH),
        gb=joblib.load(GB_MODEL_PATH),   scaler=joblib.load(SCALER_PATH),
        selector=joblib.load(SELECTOR_PATH),
        feature_names=joblib.load(FEATURE_NAMES_PATH),
        label_map=joblib.load(LABEL_MAP_PATH),
    )

# ── Preprocessing ────────────────────────────────────────────────
def preprocess_image(img_array: np.ndarray, scaler, selector) -> np.ndarray:
    raw = extract_features_from_array(img_array)
    return selector.transform(scaler.transform(raw.reshape(1, -1)))

# ── Prediction ───────────────────────────────────────────────────
MODEL_KEYS  = {"svm": "SVM", "rf": "Random Forest", "gb": "Gradient Boosting"}

def predict_single(img: np.ndarray, artifacts: Dict,
                   selected: List[str]) -> pd.DataFrame:
    X = preprocess_image(img, artifacts["scaler"], artifacts["selector"])
    rows = []
    for key in selected:
        model = artifacts[key]
        probs = model.predict_proba(X)[0]          # shape (6,)
        pred_idx  = int(np.argmax(probs))
        pred_cls  = IDX_TO_CLASS[pred_idx]
        confidence = float(probs[pred_idx])
        rows.append({
            "Model":           MODEL_KEYS[key],
            "Predicted Class": pred_cls,
            "Confidence":      confidence,
            **{cls: float(probs[i]) for i, cls in enumerate(NEU_CLASSES_SORTED)},
        })
    return pd.DataFrame(rows)

# ── Plots ────────────────────────────────────────────────────────
def fig_to_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0); return buf.read()

def make_prob_bar(probs_row: pd.Series, model_name: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 3.2), dpi=110)
    fig.patch.set_facecolor("#1A1F2E")
    ax.set_facecolor("#1A1F2E")
    classes = NEU_CLASSES_SORTED
    vals    = [probs_row.get(c, 0.0) for c in classes]
    bars = ax.barh(classes, vals, color=CLASS_COLORS, alpha=0.85, height=0.6)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Probability", color="white", fontsize=10)
    ax.set_title(f"{model_name} — Class Probabilities", color="white", fontsize=11)
    ax.tick_params(colors="white", labelsize=9)
    for sp in ax.spines.values(): sp.set_edgecolor("#2D3748")
    for bar, v in zip(bars, vals):
        ax.text(min(v + 0.01, 0.97), bar.get_y() + bar.get_height()/2,
                f"{v:.3f}", va="center", color="white", fontsize=9)
    plt.tight_layout()
    return fig

def make_cm_plot(y_true, y_pred, title) -> plt.Figure:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(N_CLASSES)))
    labels = [c[:9] for c in NEU_CLASSES_SORTED]
    fig, ax = plt.subplots(figsize=(8, 7), dpi=110)
    fig.patch.set_facecolor("#1A1F2E"); ax.set_facecolor("#1A1F2E")
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(N_CLASSES)); ax.set_yticks(range(N_CLASSES))
    ax.set_xticklabels(labels, rotation=35, ha="right", color="white", fontsize=8)
    ax.set_yticklabels(labels, color="white", fontsize=8)
    ax.set_xlabel("Predicted", color="white"); ax.set_ylabel("True", color="white")
    ax.set_title(title, color="white", fontsize=11)
    thresh = cm.max() / 2
    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=11, fontweight="bold")
    plt.tight_layout(); return fig

def make_roc_plot(y_true_arr, probs_dict) -> plt.Figure:
    from sklearn.metrics import roc_curve, auc
    y_bin = label_binarize(y_true_arr, classes=list(range(N_CLASSES)))
    model_palette = {"SVM":"#6C63FF","Random Forest":"#4ECDC4","Gradient Boosting":"#FF6B6B"}
    fig, axes = plt.subplots(1, len(probs_dict), figsize=(5*len(probs_dict), 4), dpi=110)
    if len(probs_dict) == 1: axes = [axes]
    fig.patch.set_facecolor("#1A1F2E")
    for ax, (mname, probs) in zip(axes, probs_dict.items()):
        ax.set_facecolor("#1A1F2E")
        for i, cls in enumerate(NEU_CLASSES_SORTED):
            fpr, tpr, _ = roc_curve(y_bin[:, i], probs[:, i])
            ax.plot(fpr, tpr, color=CLASS_COLORS[i], lw=1.8,
                    label=f"{cls[:8]} ({auc(fpr,tpr):.2f})")
        ax.plot([0,1],[0,1],"w--",alpha=.4,lw=1)
        ax.set_title(f"ROC OvR — {mname}", color="white", fontsize=10)
        ax.set_xlabel("FPR", color="white", fontsize=9)
        ax.set_ylabel("TPR", color="white", fontsize=9)
        ax.legend(facecolor="#1A1F2E", labelcolor="white", fontsize=7)
        ax.tick_params(colors="white")
        for sp in ax.spines.values(): sp.set_edgecolor("#2D3748")
    plt.tight_layout(); return fig

# ── Sidebar ──────────────────────────────────────────────────────
with st.sidebar:
    st.title("🔬 ManufactureGuard")
    st.caption("NEU Surface Defect · 6-Class Detection")
    st.divider()
    st.header("⚙️ Settings")
    selected_models = st.multiselect(
        "Classifiers", options=["svm","rf","gb"], default=["svm","rf","gb"],
        format_func=lambda x: {"svm":"SVM (RBF)","rf":"Random Forest",
                               "gb":"Gradient Boosting"}[x],
    )
    st.divider()
    mode = st.radio("Mode", ["🖼️ Single Image","📁 Batch Images","📊 Model Evaluation"])
    st.divider()
    st.markdown("""<div class='small-note'>
    <b>Task:</b> 6-class defect classification<br>
    <b>Features:</b> LBP · GLCM · Gabor<br>
    <b>Models:</b> SVM · RF · GBM<br>
    <b>Classes:</b><br>
    • crazing &nbsp;• inclusion<br>
    • patches &nbsp;• pitted_surface<br>
    • rolled-in_scale &nbsp;• scratches
    </div>""", unsafe_allow_html=True)

# ── Load models ──────────────────────────────────────────────────
artifacts = load_artifacts()
if "error" in artifacts:
    st.error(f"⚠️ {artifacts['error']}")
    st.info("**Steps:**\n1. Unzip `archive.zip` into `data/` folder\n"
            "2. Run `python train.py`\n3. Reload this page")
    st.stop()

if not selected_models:
    st.warning("Select at least one classifier."); st.stop()

# ── Header ────────────────────────────────────────────────────────
st.title("🔬 ManufactureGuard — Surface Defect Detection")
st.caption("6-class multiclass classification: crazing · inclusion · patches · "
           "pitted_surface · rolled-in_scale · scratches")

cols = st.columns(3)
for col, (k, lbl, desc) in zip(cols, [
    ("svm","🔵 SVM (RBF)","One-vs-rest multiclass"),
    ("rf","🌲 Random Forest","400 trees · feature importance"),
    ("gb","🚀 Gradient Boosting","300 estimators"),
]):
    col.markdown(f"<div class='metric-card'><h4>{lbl}</h4><p>{desc}</p></div>",
                 unsafe_allow_html=True)
st.divider()

# ════════════════════════════════════════════════════════
# MODE 1 — Single Image
# ════════════════════════════════════════════════════════
if "Single" in mode:
    st.subheader("🖼️ Single Image Inspection")
    uploaded = st.file_uploader("Upload a surface image",
                                type=["jpg","jpeg","png","bmp","tif","tiff"])
    true_class = st.selectbox("True defect class (optional)",
                              ["Unknown"] + NEU_CLASSES_SORTED)

    if uploaded:
        fb = np.frombuffer(uploaded.read(), np.uint8)
        img_bgr = cv2.imdecode(fb, cv2.IMREAD_COLOR)
        if img_bgr is None:
            st.error("Cannot decode image."); st.stop()

        col_img, col_res = st.columns([1, 1.5])
        with col_img:
            st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB),
                     caption="Uploaded image", use_container_width=True)
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            st.image(cv2.resize(gray, IMG_SIZE),
                     caption=f"Grayscale {IMG_SIZE[0]}×{IMG_SIZE[1]}",
                     use_container_width=True, channels="GRAY")

        with col_res:
            with st.spinner("Extracting texture features …"):
                t0 = time.time()
                res_df = predict_single(img_bgr, artifacts, selected_models)
                elapsed = time.time() - t0
            st.caption(f"⏱ {elapsed*1000:.0f} ms")

            # Majority vote on predicted class
            verdict = res_df["Predicted Class"].mode()[0]
            color, bg = CLASS_BADGE_COLORS.get(verdict, ("#fff","#222"))
            st.markdown(
                f"<div style='text-align:center;margin:10px 0'>"
                f"<span class='cls-badge' style='color:{color};"
                f"background:{bg};border:1px solid {color}55'>"
                f"🏷️ {verdict.upper()}</span></div>",
                unsafe_allow_html=True)

            if true_class != "Unknown":
                correct = verdict == true_class
                (st.success if correct else st.error)(
                    f"{'✅ Correct' if correct else '❌ Wrong'} — true class: {true_class}")

            # Per-model confidence table
            st.dataframe(
                res_df[["Model","Predicted Class","Confidence"]].style.format(
                    {"Confidence":"{:.4f}"}),
                use_container_width=True)

            # Probability bars for each model
            for _, row in res_df.iterrows():
                fig = make_prob_bar(row, row["Model"])
                st.pyplot(fig, use_container_width=True)
                plt.close("all")

# ════════════════════════════════════════════════════════
# MODE 2 — Batch Images
# ════════════════════════════════════════════════════════
elif "Batch" in mode:
    st.subheader("📁 Batch Image Inspection")
    st.info("Name files with class prefix (e.g. `crazing_001.jpg`) for auto-accuracy.")
    files = st.file_uploader("Upload images", type=["jpg","jpeg","png","bmp"],
                             accept_multiple_files=True)

    if files and st.button("🔍 Run Batch Detection", type="primary"):
        rows, prog = [], st.progress(0, "Processing …")
        for idx, uf in enumerate(files):
            fb = np.frombuffer(uf.read(), np.uint8)
            img = cv2.imdecode(fb, cv2.IMREAD_COLOR)
            if img is None: continue

            true_cls = next((c for c in NEU_CLASSES_SORTED
                             if c.replace("-","_") in uf.name.lower().replace("-","_")),
                            "unknown")
            try:
                res = predict_single(img, artifacts, selected_models)
            except Exception as e:
                st.warning(f"{uf.name}: {e}"); continue

            verdict = res["Predicted Class"].mode()[0]
            row = {"Image": uf.name, "True Class": true_cls, "Verdict": verdict}
            row["Correct"] = ("✅" if verdict == true_cls else "❌") \
                if true_cls != "unknown" else "—"
            for _, r in res.iterrows():
                row[f"{r['Model']} Pred"] = r["Predicted Class"]
                row[f"{r['Model']} Conf"] = round(r["Confidence"], 4)
            rows.append(row)
            prog.progress((idx+1)/len(files), f"{idx+1}/{len(files)}")

        prog.empty()
        if not rows: st.error("No images processed."); st.stop()

        result_df = pd.DataFrame(rows)
        labeled = result_df[result_df["True Class"] != "unknown"]
        c1,c2,c3 = st.columns(3)
        c1.metric("Total", len(result_df))
        if len(labeled):
            c2.metric("Correct", int((labeled["Correct"]=="✅").sum()))
            c3.metric("Accuracy", f"{(labeled['Correct']=='✅').mean():.1%}")

        st.dataframe(result_df, use_container_width=True, height=400)
        st.download_button("⬇️ Download CSV",
            data=result_df.to_csv(index=False).encode(),
            file_name="defect_predictions.csv", mime="text/csv")

# ════════════════════════════════════════════════════════
# MODE 3 — Evaluation Dashboard
# ════════════════════════════════════════════════════════
else:
    st.subheader("📊 Model Evaluation Dashboard")
    st.markdown("Upload labelled images. **File names must contain the class name** "
                "(e.g. `crazing_001.jpg`, `patches_099.jpg`).")
    eval_files = st.file_uploader("Upload labelled images",
                                  type=["jpg","jpeg","png","bmp"],
                                  accept_multiple_files=True)

    if eval_files and st.button("📊 Run Evaluation", type="primary"):
        imgs, y_true, cls_names = [], [], []
        prog = st.progress(0, "Loading …")
        for idx, uf in enumerate(eval_files):
            cls = next((c for c in NEU_CLASSES_SORTED
                        if c.replace("-","_") in uf.name.lower().replace("-","_")),
                       None)
            if cls is None: continue
            fb = np.frombuffer(uf.read(), np.uint8)
            img = cv2.imdecode(fb, cv2.IMREAD_COLOR)
            if img is None: continue
            imgs.append(img); y_true.append(CLASS_TO_IDX[cls]); cls_names.append(cls)
            prog.progress((idx+1)/len(eval_files))
        prog.empty()

        if len(imgs) < N_CLASSES:
            st.error(f"Need images from all {N_CLASSES} classes."); st.stop()

        y_true_arr = np.array(y_true, dtype=int)
        st.success(f"Evaluating on {len(imgs)} images across "
                   f"{len(set(y_true))} classes.")

        with st.spinner("Extracting texture features …"):
            p2 = st.progress(0)
            feats = []
            for i, img in enumerate(imgs):
                feats.append(extract_features_from_array(img))
                p2.progress((i+1)/len(imgs))
            p2.empty()

        X = artifacts["selector"].transform(
            artifacts["scaler"].transform(np.vstack(feats)))

        # Metrics
        metrics_rows, probs_dict = [], {}
        for key in selected_models:
            lbl   = MODEL_KEYS[key]
            model = artifacts[key]
            y_pred = model.predict(X)
            y_prob = model.predict_proba(X)
            probs_dict[lbl] = y_prob

            y_bin = label_binarize(y_true_arr, classes=list(range(N_CLASSES)))
            try:
                auc_ovr = roc_auc_score(y_bin, y_prob, multi_class="ovr", average="macro")
            except Exception:
                auc_ovr = float("nan")

            metrics_rows.append({
                "Model":            lbl,
                "Accuracy":         round(accuracy_score(y_true_arr, y_pred), 4),
                "Macro F1":         round(f1_score(y_true_arr, y_pred, average="macro",
                                                   zero_division=0), 4),
                "Macro Precision":  round(precision_score(y_true_arr, y_pred,
                                                          average="macro",
                                                          zero_division=0), 4),
                "Macro Recall":     round(recall_score(y_true_arr, y_pred,
                                                       average="macro",
                                                       zero_division=0), 4),
                "ROC-AUC (OvR)":    round(auc_ovr, 4),
            })

        st.subheader("📈 Performance Metrics")
        mdf = pd.DataFrame(metrics_rows).set_index("Model")
        st.dataframe(mdf.style.highlight_max(axis=0, color="#6C63FF33",
            subset=["Accuracy","Macro F1","ROC-AUC (OvR)"]),
            use_container_width=True)

        # ROC curves
        st.subheader("ROC Curves (One-vs-Rest)")
        fig_roc = make_roc_plot(y_true_arr, probs_dict)
        st.pyplot(fig_roc, use_container_width=True); plt.close("all")

        # Confusion matrices
        st.subheader("Confusion Matrices")
        cm_cols = st.columns(len(selected_models))
        for i, key in enumerate(selected_models):
            lbl = MODEL_KEYS[key]
            y_pred = artifacts[key].predict(X)
            with cm_cols[i]:
                fig_cm = make_cm_plot(y_true_arr, y_pred, lbl)
                st.pyplot(fig_cm, use_container_width=True); plt.close("all")

        # Per-class report
        st.subheader("📋 Per-Class Report (Random Forest)")
        if "rf" in selected_models:
            y_pred_rf = artifacts["rf"].predict(X)
            report = classification_report(y_true_arr, y_pred_rf,
                                           target_names=NEU_CLASSES_SORTED,
                                           zero_division=0, output_dict=True)
            report_df = pd.DataFrame(report).T.drop(
                columns=["support"], errors="ignore").round(4)
            st.dataframe(report_df, use_container_width=True)

        # Feature importances
        st.subheader("🔍 Feature Importances")
        feat_names = artifacts["feature_names"]
        fi_cols = st.columns(len(selected_models))
        for i, key in enumerate(selected_models):
            lbl = MODEL_KEYS[key]; model = artifacts[key]
            with fi_cols[i]:
                st.markdown(f"**{lbl}**")
                if hasattr(model, "feature_importances_"):
                    imp = model.feature_importances_
                    n = min(15, len(imp)); idx = np.argsort(imp)[-n:][::-1]
                    st.dataframe(pd.DataFrame({
                        "Feature":    [feat_names[j] if j < len(feat_names)
                                       else f"f{j}" for j in idx],
                        "Importance": imp[idx].round(5),
                    }), use_container_width=True, height=300)
                else:
                    st.caption("Not available for SVM/RBF.")

st.divider()
st.markdown("<div class='small-note' style='text-align:center'>"
            "ManufactureGuard · NEU-DET · LBP+GLCM+Gabor · SVM·RF·GBM · 6-class</div>",
            unsafe_allow_html=True)
