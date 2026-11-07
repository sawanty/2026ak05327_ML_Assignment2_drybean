# Dry Bean Variety Classification — ML Assignment 2

**Student:** Yatin Sawant · **BITS ID:** 2026ak05327 · M.Tech (AIML), BITS Pilani WILP

| | Link |
|---|---|
| GitHub repository | https://github.com/sawanty/2026ak05327_ML_Assignment2_drybean |
| Live Streamlit app | `<paste your *.streamlit.app link here>` |

---

## a. Problem statement

A seed company wants to sort harvested dry beans by variety automatically instead of by hand.
Each grain is photographed and 16 shape measurements are extracted from the image.
The task is to **predict which of 7 registered dry bean varieties a grain belongs to** from those
16 measurements. This is a **multi-class classification** problem.

Five classifiers (Logistic Regression, Decision Tree, K-Nearest Neighbours, Gaussian Naive Bayes
and Support Vector Machine) are trained on the same data, compared on six metrics, and served
through an interactive Streamlit app that scores the saved models on an uploaded test CSV.

## b. Dataset description

| Item | Value |
|---|---|
| Name | Dry Bean Dataset |
| Source | UCI Machine Learning Repository, id 602 — https://archive.ics.uci.edu/dataset/602/dry+bean+dataset |
| Authors | M. Koklu & I. A. Ozkan (2020), *Computers and Electronics in Agriculture* 174, 105507 |
| Instances | 13,611 (13,543 after removing 68 exact duplicate rows) |
| Features | 16 numeric (12 dimension features + 4 shape factors) |
| Target | `Class` — 7 varieties |
| Missing values | None |

**Features**

| Group | Columns |
|---|---|
| Size | `Area`, `Perimeter`, `ConvexArea`, `EquivDiameter` |
| Axes | `MajorAxisLength`, `MinorAxisLength`, `AspectRation`, `Eccentricity` |
| Shape | `Extent`, `Solidity`, `roundness`, `Compactness` |
| Shape factors | `ShapeFactor1`, `ShapeFactor2`, `ShapeFactor3`, `ShapeFactor4` |

**Class distribution (full data)** — the classes are imbalanced:

| DERMASON | SIRA | SEKER | HOROZ | CALI | BARBUNYA | BOMBAY |
|---|---|---|---|---|---|---|
| 3,546 | 2,636 | 2,027 | 1,928 | 1,630 | 1,322 | 522 |

Several features are almost copies of each other (for example `Area` vs `ConvexArea`,
`Compactness` vs `ShapeFactor3`, absolute correlation ≥ 0.99), and their scales differ by
up to six orders of magnitude (`Area` ≈ 50,000 vs `ShapeFactor2` ≈ 0.001).

**Preparation**

1. Removed 68 duplicate rows.
2. Stratified 80 / 20 split (seed 2026): **10,834 training rows** and **2,709 test rows**.
   The test rows are saved as [`test_data.csv`](test_data.csv) and are never used for training or tuning.
3. Labels are encoded with a `LabelEncoder` (saved in `model/label_encoder.joblib`).
4. Every model except the Decision Tree is wrapped in a `Pipeline` with a `StandardScaler`, so
   the saved `.joblib` file contains the scaler fitted on training data only.

## c. GitHub repository

https://github.com/sawanty/2026ak05327_ML_Assignment2_drybean

```
2026ak05327_ML_Assignment2_drybean/
├── app.py                     # Streamlit app (loads saved models, no retraining)
├── requirements.txt
├── README.md
├── test_data.csv              # held-out 20% test split (features + Class)
├── data/
│   └── dry_bean.csv           # full UCI dataset converted from ARFF to CSV
└── model/
    ├── train_models.py        # training, tuning, evaluation, saving
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    ├── svm.joblib
    ├── label_encoder.joblib
    └── metrics.json           # best hyper-parameters + test metrics
```

**Reproduce**

```bash
pip install -r requirements.txt
python model/train_models.py   # retrains, rewrites model/*.joblib and test_data.csv
streamlit run app.py
```

## d. Models used

Each model was tuned with `GridSearchCV` (5-fold stratified CV on the training rows,
scoring = macro F1), then evaluated once on the held-out test set.

| Model | Search space | Selected |
|---|---|---|
| Logistic Regression | C ∈ {0.1, 1, 10, 100} | C = 1 |
| Decision Tree | max_depth ∈ {6, 8, 10, 12, None}, min_samples_leaf ∈ {1, 5, 10}, criterion ∈ {gini, entropy} | gini, depth 10, leaf 10 |
| KNN | k ∈ {5, 9, 15, 21}, weights ∈ {uniform, distance} | k = 9, distance |
| Naive Bayes (Gaussian) | var_smoothing ∈ {1e-9, 1e-7, 1e-5, 1e-3} | 1e-3 |
| SVM (RBF kernel) | C ∈ {1, 10, 50}, gamma ∈ {scale, 0.05} | C = 10, gamma = scale |

### Comparison table (test set, 2,709 rows)

Precision, Recall and F1 are macro-averaged over the 7 classes; AUC is one-vs-rest macro AUC.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9251 | **0.9947** | 0.9373 | 0.9375 | 0.9374 | 0.9094 |
| Decision Tree | 0.9110 | 0.9821 | 0.9252 | 0.9232 | 0.9241 | 0.8924 |
| KNN | 0.9228 | 0.9882 | 0.9374 | 0.9364 | 0.9369 | 0.9067 |
| Naive Bayes | 0.8992 | 0.9914 | 0.9067 | 0.9123 | 0.9092 | 0.8786 |
| Support Vector Machine | **0.9280** | 0.9946 | **0.9404** | **0.9398** | **0.9400** | **0.9129** |

Per-class F1 on the test set:

| Model | BARBUNYA | BOMBAY | CALI | DERMASON | HOROZ | SEKER | SIRA |
|---|---|---|---|---|---|---|---|
| Logistic Regression | 0.934 | 1.000 | 0.941 | 0.919 | 0.953 | 0.945 | 0.870 |
| Decision Tree | 0.912 | 0.995 | 0.931 | 0.906 | 0.941 | 0.929 | 0.853 |
| KNN | 0.939 | 1.000 | 0.948 | 0.911 | 0.956 | 0.945 | 0.860 |
| Naive Bayes | 0.864 | 0.990 | 0.887 | 0.902 | 0.938 | 0.936 | 0.847 |
| SVM | 0.933 | 1.000 | 0.947 | 0.918 | 0.954 | 0.954 | 0.875 |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Very strong for a linear model: 92.5 % accuracy and the highest AUC (0.9947). After scaling, most bean varieties are close to linearly separable in this shape space. Moderate regularisation (C = 1) was best; larger C did not help, which suggests the remaining errors are overlapping classes rather than under-fitting. Its main weakness is the SIRA ↔ DERMASON border (92 of its ~200 errors). |
| Decision Tree | Lowest AUC (0.982) and second-lowest accuracy (91.1 %). Without depth limits it over-fits; CV picked depth 10 with ≥ 10 samples per leaf (195 leaves). Its probabilities are coarse leaf frequencies, which is why its AUC trails the others even though its accuracy is close. It mostly splits on `ShapeFactor1`, `Compactness` and `MajorAxisLength`, so it is the easiest model to explain, but it makes the most SIRA → DERMASON mistakes (56). |
| KNN | Close to the top (92.3 % accuracy, F1 0.937) once features are scaled; without scaling `Area` would dominate the distance. Distance-weighted voting with k = 9 worked best. It has the best F1 on BARBUNYA, CALI and HOROZ, but a lower AUC (0.988) than LR/SVM because its probability estimates come from only 9 neighbours. Prediction is slower and the saved model stores the whole training set (≈ 1.3 MB). |
| Naive Bayes | Weakest on every metric except AUC (accuracy 89.9 %, MCC 0.879). Gaussian NB assumes the features are independent, but this dataset has groups of features with correlation ≥ 0.99, so the same evidence is counted several times. It confuses BARBUNYA with CALI (29 errors) much more than the other models do. Its AUC is still high (0.991): the ranking of classes is good even when the top choice is wrong. A larger `var_smoothing` (1e-3) helped. |
| Support Vector Machine | Best accuracy (92.8 %), macro F1 (0.940) and MCC (0.913), and AUC equal to LR within 0.0001. The RBF kernel draws slightly curved class borders that fit the SIRA / DERMASON / SEKER region a little better than a straight line. It was also best in cross-validation (CV F1 0.942), so the win is not just luck on the test split. It is the slowest model to tune (~50 s). |
| **Overall winner** | **Support Vector Machine (RBF, C = 10).** It leads on 4 of 6 metrics and ties on AUC. The gap to Logistic Regression is small (≈ 0.3 points of accuracy), so LR is a good, faster and more interpretable fallback. All models get BOMBAY almost perfect (it is much larger than the others) and all struggle most with SIRA vs DERMASON, which look very similar in size and shape — this overlap, not model choice, sets the ~92–93 % ceiling. |

## Streamlit app

[`app.py`](app.py) provides:

- **CSV upload** of test data (must contain the 16 feature columns and `Class`), plus a button to
  download the bundled `test_data.csv` and a toggle to use it directly.
- **Model selection dropdown** for the five saved models (shows the tuned hyper-parameters).
- **Evaluation metrics**: Accuracy, AUC, Precision, Recall, F1, MCC for the chosen model.
- **Confusion matrix** (counts or row percentages) and **classification report**.
- Prediction table with download, and a side-by-side comparison of all five models on the uploaded file.

The app only loads `model/*.joblib`; it never refits a model on the uploaded data.
