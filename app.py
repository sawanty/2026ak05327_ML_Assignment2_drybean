"""
Dry Bean Classifier - Streamlit app for ML Assignment 2.

Upload a test CSV (16 bean shape features + the "Class" column), pick one of the
five pre-trained models and see its metrics, confusion matrix and classification
report. Models are only loaded from model/*.joblib; nothing is retrained here.
"""

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
SAMPLE_TEST = ROOT / "test_data.csv"

st.set_page_config(page_title="Dry Bean Classifier", page_icon="🫘", layout="wide")


# ---------- loading (cached so models are read from disk once) ----------
@st.cache_resource
def load_artifacts():
    meta = json.loads((MODEL_DIR / "metrics.json").read_text())
    encoder = joblib.load(MODEL_DIR / "label_encoder.joblib")
    models = {name: joblib.load(MODEL_DIR / info["file"]) for name, info in meta["models"].items()}
    return meta, encoder, models


def safe_auc(y_true_enc, y_proba):
    """One-vs-rest macro AUC over the classes present in the file (needs >= 2 classes)."""
    present = np.unique(y_true_enc)
    if len(present) < 2:
        return float("nan")
    if len(present) == 2:
        return roc_auc_score(y_true_enc == present[1], y_proba[:, present[1]])
    sub = y_proba[:, present]
    sub = sub / sub.sum(axis=1, keepdims=True)
    return roc_auc_score(y_true_enc, sub, multi_class="ovr", average="macro", labels=present)


def score_model(model, X, y_true_enc):
    y_pred = model.predict(X)
    auc = safe_auc(y_true_enc, model.predict_proba(X))
    metrics = {
        "Accuracy": accuracy_score(y_true_enc, y_pred),
        "AUC": auc,
        "Precision": precision_score(y_true_enc, y_pred, average="macro", zero_division=0),
        "Recall": recall_score(y_true_enc, y_pred, average="macro", zero_division=0),
        "F1": f1_score(y_true_enc, y_pred, average="macro", zero_division=0),
        "MCC": matthews_corrcoef(y_true_enc, y_pred),
    }
    return y_pred, metrics


meta, encoder, models = load_artifacts()
FEATURES = meta["features"]
TARGET = meta["target"]
CLASSES = list(encoder.classes_)

# ---------- sidebar ----------
with st.sidebar:
    st.header("🫘 Controls")
    model_name = st.selectbox("Choose a model", list(models.keys()), index=4)
    info = meta["models"][model_name]
    st.caption("Tuned hyper-parameters (5-fold CV on training data):")
    st.json(info["best_params"])

    st.divider()
    st.markdown("**Need a test file?**")
    with open(SAMPLE_TEST, "rb") as f:
        st.download_button(
            "Download test_data.csv", f, file_name="test_data.csv", mime="text/csv"
        )
    st.caption(f"Held-out 20% split · {meta['test_rows']} rows · never seen in training")

# ---------- header ----------
st.title("Dry Bean Variety Classifier")
st.markdown(
    "Seven dry bean varieties (UCI Dry Bean dataset, 16 shape features from grain images). "
    "Upload a **test CSV**, choose a model in the sidebar, and the app scores the saved model "
    "on your file. **No retraining happens here.**"
)

uploaded = st.file_uploader("Upload test CSV (features + 'Class' column)", type=["csv"])
use_sample = st.toggle("Or use the bundled test_data.csv", value=uploaded is None)

if uploaded is not None:
    data = pd.read_csv(uploaded)
    source = uploaded.name
elif use_sample:
    data = pd.read_csv(SAMPLE_TEST)
    source = "test_data.csv (bundled)"
else:
    st.info("Upload a CSV file or switch on the bundled test data to begin.")
    st.stop()

# ---------- validate columns ----------
missing = [c for c in FEATURES + [TARGET] if c not in data.columns]
if missing:
    st.error(f"The file is missing these required columns: {', '.join(missing)}")
    st.stop()

data = data.dropna(subset=FEATURES + [TARGET])
unknown = sorted(set(data[TARGET].astype(str).str.strip().str.upper()) - set(CLASSES))
if unknown:
    st.error(f"Unknown class labels in '{TARGET}': {unknown}. Expected one of {CLASSES}.")
    st.stop()

X = data[FEATURES]
y_true = encoder.transform(data[TARGET].astype(str).str.strip().str.upper())

with st.expander(f"Preview of {source} · {len(data)} rows", expanded=False):
    st.dataframe(data.head(20), width="stretch")
    st.bar_chart(data[TARGET].value_counts())

# ---------- selected model ----------
y_pred, m = score_model(models[model_name], X, y_true)

st.subheader(f"📊 {model_name} — evaluation metrics")
cols = st.columns(6)
for col, (k, v) in zip(cols, m.items()):
    col.metric(k, "n/a" if np.isnan(v) else f"{v:.4f}")
st.caption("Precision, Recall and F1 are macro-averaged; AUC is one-vs-rest macro.")

tab_cm, tab_rep, tab_pred = st.tabs(["Confusion matrix", "Classification report", "Predictions"])

with tab_cm:
    labels_idx = np.arange(len(CLASSES))
    cm = confusion_matrix(y_true, y_pred, labels=labels_idx)
    normalise = st.checkbox("Show row percentages", value=False)
    shown = cm / cm.sum(axis=1, keepdims=True).clip(min=1) if normalise else cm
    fig, ax = plt.subplots(figsize=(7, 5.5))
    sns.heatmap(
        shown, annot=True, fmt=".2f" if normalise else "d", cmap="YlOrBr",
        xticklabels=CLASSES, yticklabels=CLASSES, cbar=False, ax=ax,
    )
    ax.set_xlabel("Predicted variety")
    ax.set_ylabel("True variety")
    ax.set_title(f"{model_name}")
    plt.tight_layout()
    st.pyplot(fig, width="content")

with tab_rep:
    rep = classification_report(
        y_true, y_pred, labels=labels_idx, target_names=CLASSES, output_dict=True, zero_division=0
    )
    rep_df = pd.DataFrame(rep).T
    rep_df["support"] = rep_df["support"].astype(int)
    st.dataframe(rep_df.style.format({"precision": "{:.4f}", "recall": "{:.4f}",
                                      "f1-score": "{:.4f}"}), width="stretch")

with tab_pred:
    out = data.copy()
    out["Predicted"] = encoder.inverse_transform(y_pred)
    out["Correct"] = out["Predicted"] == out[TARGET].astype(str).str.strip().str.upper()
    st.dataframe(out[[TARGET, "Predicted", "Correct"] + FEATURES].head(200), width="stretch")
    st.download_button(
        "Download predictions CSV", out.to_csv(index=False).encode(),
        file_name=f"predictions_{model_name.lower().replace(' ', '_')}.csv", mime="text/csv",
    )

# ---------- all models side by side ----------
st.subheader("🏁 All five models on this file")
rows = {name: score_model(mdl, X, y_true)[1] for name, mdl in models.items()}
comp = pd.DataFrame(rows).T
st.dataframe(
    comp.style.format("{:.4f}").highlight_max(axis=0, color="#f6d58e"),
    width="stretch",
)
best = comp["F1"].idxmax()
st.success(f"Best macro-F1 on this file: **{best}** ({comp.loc[best, 'F1']:.4f})")

st.caption("ML Assignment 2 · M.Tech AIML · BITS Pilani WILP")
