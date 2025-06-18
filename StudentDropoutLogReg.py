"""
Early-Term Dropout-Risk Model (Logistic Regression)
--------------------------------------------------
• Dataset : UCI Student Performance (math + Portuguese)
• Label   : 1 if (G3 < 10) OR (absences > 30)
• Features: early-term only (no G2, no G3)
"""

import pathlib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing  import StandardScaler, OneHotEncoder
from sklearn.compose        import ColumnTransformer
from sklearn.pipeline       import Pipeline
from sklearn.linear_model   import LogisticRegression
from sklearn.metrics        import (classification_report,
                                    confusion_matrix,
                                    roc_auc_score,
                                    RocCurveDisplay)

# ───────────────────────────────────────────────────────────
# 1. Load data and build label
# ───────────────────────────────────────────────────────────
DATA_DIR = pathlib.Path(".")
df = (
    pd.read_csv(DATA_DIR / "student-mat.csv", sep=";")
      .pipe(lambda d: pd.concat([d,
                                 pd.read_csv(DATA_DIR / "student-por.csv", sep=";")],
                                ignore_index=True))
)

df["dropped_out"] = ((df["G3"] < 10) | (df["absences"] > 30)).astype(int)

# ───────────────────────────────────────────────────────────
# 2. Define X, y   (drop late-term grades)
# ───────────────────────────────────────────────────────────
X = df.drop(columns=["dropped_out", "G2", "G3"])
y = df["dropped_out"]

numeric_features      = ["G1", "absences", "studytime", "failures"]
categorical_features  = [c for c in X.columns if c not in numeric_features]

# ───────────────────────────────────────────────────────────
# 3. Pre-processing pipeline
# ───────────────────────────────────────────────────────────
preprocess = ColumnTransformer(
    [("num", Pipeline([("scaler", StandardScaler())]), numeric_features),
     ("cat", Pipeline([("onehot", OneHotEncoder(handle_unknown="ignore"))]),
      categorical_features)],
    remainder="drop",
    verbose_feature_names_out=False,
)

# ───────────────────────────────────────────────────────────
# 4. Logistic Regression + hyper-parameter grid
# ───────────────────────────────────────────────────────────
logreg = LogisticRegression(
    max_iter=1000,
    solver="liblinear",        # supports L1 + L2
    class_weight="balanced"    # handle imbalance
)

param_grid = {
    "clf__C":       [0.01, 0.1, 1, 10],
    "clf__penalty": ["l1", "l2"]
}

pipe = Pipeline([("prep", preprocess), ("clf", logreg)])

grid = GridSearchCV(pipe,
                    param_grid,
                    cv=5,
                    scoring="roc_auc",
                    n_jobs=-1,
                    verbose=1)

# ───────────────────────────────────────────────────────────
# 5. Train / test split & fit
# ───────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.20, random_state=42
)

grid.fit(X_train, y_train)
best_model = grid.best_estimator_

# ───────────────────────────────────────────────────────────
# 6. Evaluation  (threshold tuned to 0.35)
# ───────────────────────────────────────────────────────────
y_prob = best_model.predict_proba(X_test)[:, 1]
y_pred = (y_prob >= 0.35).astype(int)

auc = roc_auc_score(y_test, y_prob)

print("\n=== Best hyper-parameters ===")
print(grid.best_params_)

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred))

print("Confusion Matrix")
print(confusion_matrix(y_test, y_pred))

print(f"ROC-AUC: {auc:.3f}")

# ───────────────────────────────────────────────────────────
# 7. Inspect coefficients (sorted by magnitude)
# ───────────────────────────────────────────────────────────
feature_names = best_model.named_steps["prep"].get_feature_names_out()
coeffs        = best_model.named_steps["clf"].coef_[0]
idx           = np.argsort(np.abs(coeffs))[::-1][:15]

print("\n=== Top 15 Coefficients ===")
for rank, i in enumerate(idx, start=1):
    sign = "+" if coeffs[i] >= 0 else "-"
    print(f"{rank:2d}. {feature_names[i]:<25s} {sign}{abs(coeffs[i]):.4f}")

# Optional: ROC curve
try:
    RocCurveDisplay.from_predictions(y_test, y_prob)
except Exception:
    pass
