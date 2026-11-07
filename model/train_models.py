"""
Train the five classifiers for ML Assignment 2 on the UCI Dry Bean dataset.

Run from the project root:
    python model/train_models.py

What it does:
1. Loads data/dry_bean.csv and drops duplicate rows.
2. Makes a stratified 80/20 train/test split. The test part is written to
   test_data.csv so it can be uploaded to the Streamlit app.
3. Tunes each model with a small grid search (5-fold CV on the training part only).
4. Evaluates each tuned model on the held-out test part with six metrics.
5. Saves every fitted pipeline (scaler + model) as a .joblib file in model/,
   plus the label encoder and a metrics summary.
"""

import json
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "dry_bean.csv"
TEST_FILE = ROOT / "test_data.csv"
MODEL_DIR = ROOT / "model"

TARGET = "Class"
SEED = 2026  # fixed seed so results can be reproduced

# scikit-learn 1.9 marks SVC(probability=True) as deprecated; it still works the same.
warnings.filterwarnings("ignore", message=".*probability.*", category=FutureWarning)


def bean_pipeline(estimator, scale=True):
    """Wrap a classifier with a StandardScaler so the saved file is self-contained."""
    steps = [("scaler", StandardScaler())] if scale else []
    steps.append(("clf", estimator))
    return Pipeline(steps)


# Model name -> (pipeline, parameter grid, file name)
MODEL_SPECS = {
    "Logistic Regression": (
        bean_pipeline(LogisticRegression(max_iter=3000)),
        {"clf__C": [0.1, 1.0, 10.0, 100.0]},
        "logistic_regression.joblib",
    ),
    "Decision Tree": (
        # Trees do not need scaling, so the scaler is skipped here.
        bean_pipeline(DecisionTreeClassifier(random_state=SEED), scale=False),
        {
            "clf__max_depth": [6, 8, 10, 12, None],
            "clf__min_samples_leaf": [1, 5, 10],
            "clf__criterion": ["gini", "entropy"],
        },
        "decision_tree.joblib",
    ),
    "KNN": (
        bean_pipeline(KNeighborsClassifier()),
        {"clf__n_neighbors": [5, 9, 15, 21], "clf__weights": ["uniform", "distance"]},
        "knn.joblib",
    ),
    "Naive Bayes": (
        bean_pipeline(GaussianNB()),
        {"clf__var_smoothing": [1e-9, 1e-7, 1e-5, 1e-3]},
        "naive_bayes.joblib",
    ),
    "Support Vector Machine": (
        bean_pipeline(SVC(kernel="rbf", probability=True, random_state=SEED)),
        {"clf__C": [1.0, 10.0, 50.0], "clf__gamma": ["scale", 0.05]},
        "svm.joblib",
    ),
}


def compute_metrics(y_true, y_pred, y_proba):
    """Six required metrics. Multi-class scores use macro averaging; AUC is one-vs-rest."""
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro"),
        "Precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "Recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "F1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


def main():
    df = pd.read_csv(DATA_FILE)
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"Loaded {before} rows, {before - len(df)} duplicates removed -> {len(df)} rows")

    feature_cols = [c for c in df.columns if c != TARGET]

    train_df, test_df = train_test_split(
        df, test_size=0.20, stratify=df[TARGET], random_state=SEED
    )
    test_df.to_csv(TEST_FILE, index=False)
    print(f"Train rows: {len(train_df)} | Test rows: {len(test_df)} (saved to {TEST_FILE.name})")

    encoder = LabelEncoder().fit(df[TARGET])
    X_train, y_train = train_df[feature_cols], encoder.transform(train_df[TARGET])
    X_test, y_test = test_df[feature_cols], encoder.transform(test_df[TARGET])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    summary = {}

    for name, (pipe, grid, fname) in MODEL_SPECS.items():
        start = time.time()
        search = GridSearchCV(pipe, grid, scoring="f1_macro", cv=cv, n_jobs=-1)
        search.fit(X_train, y_train)
        best = search.best_estimator_

        y_pred = best.predict(X_test)
        y_proba = best.predict_proba(X_test)
        scores = compute_metrics(y_test, y_pred, y_proba)

        joblib.dump(best, MODEL_DIR / fname, compress=3)
        summary[name] = {
            "file": fname,
            "best_params": {k.replace("clf__", ""): v for k, v in search.best_params_.items()},
            "cv_f1_macro": round(float(search.best_score_), 4),
            "test_metrics": {k: round(float(v), 4) for k, v in scores.items()},
        }
        print(f"{name:<24} {time.time() - start:6.1f}s  best={summary[name]['best_params']}")
        print("   " + "  ".join(f"{k}={v:.4f}" for k, v in scores.items()))

    joblib.dump(encoder, MODEL_DIR / "label_encoder.joblib")
    meta = {
        "dataset": "UCI Dry Bean Dataset (id 602)",
        "target": TARGET,
        "features": feature_cols,
        "classes": encoder.classes_.tolist(),
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "seed": SEED,
        "models": summary,
    }
    (MODEL_DIR / "metrics.json").write_text(json.dumps(meta, indent=2))

    table = pd.DataFrame({m: v["test_metrics"] for m, v in summary.items()}).T
    print("\nComparison table (test set):")
    print(table.to_string(float_format="{:.4f}".format))


if __name__ == "__main__":
    np.random.seed(SEED)
    main()
