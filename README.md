# 🎓 Student Dropout Risk Prediction (Logistic Regression)

This project builds a machine learning model to predict early student dropout risk using logistic regression. It focuses on early-term indicators only, making it ideal for proactive interventions.

---

## 📁 Dataset

- **Source**: [UCI Student Performance Dataset](https://archive.ics.uci.edu/ml/datasets/student+performance)
- **Files used**: 
  - `student-mat.csv` (Math students)
  - `student-por.csv` (Portuguese students)
- **Label Definition**: A student is considered at risk if:
  - `G3` (final grade) < 10 **OR**
  - `absences` > 30

---

## 🧠 Model Overview

- **Model**: Logistic Regression
- **Features**:
  - Early-term only (no `G2`, no `G3`)
  - Numerical: `G1`, `absences`, `studytime`, `failures`
  - Categorical: all others
- **Label**: `dropped_out` (binary)

---

## ⚙️ Pipeline Steps

1. **Data Merging**: Math + Portuguese datasets
2. **Feature Selection**: Drop late-term grades
3. **Preprocessing**:
   - Standard scaling (numerical)
   - One-hot encoding (categorical)
4. **Modeling**:
   - Logistic Regression with `class_weight='balanced'`
   - Hyperparameter tuning via `GridSearchCV`
   - ROC-AUC as scoring metric
5. **Threshold Tuning**:
   - Predictions made using a **0.35** probability threshold (instead of default 0.5)
6. **Interpretability**:
   - Top 15 coefficients printed

---

## 📊 Metrics

- **ROC-AUC**: Reported at evaluation
- **Classification Report**: Includes precision, recall, and F1-score
- **Confusion Matrix**: Displayed for threshold = 0.35
- **Top Predictors**: Sorted coefficients shown for interpretation

---

## 📈 Visualization

- Optional ROC Curve plotted using `RocCurveDisplay`

---

## 🛠️ Requirements

```bash
pip install pandas scikit-learn matplotlib
```

If using `uv`:
```bash
uv pip install pandas scikit-learn matplotlib
```

---

## 🚀 How to Run

```bash
python StudentDropoutLogReg.py
```

---

## 🧾 License

This project is for educational use only. UCI dataset license applies.

---

## 🤖 Author

Built with ❤️ using `scikit-learn`, `pandas`, and `GridSearchCV` by [techhound](https://github.com/techhound).
