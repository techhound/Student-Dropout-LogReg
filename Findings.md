

Author: James CochraneDate: June 17, 2025Project Type: Predictive Analytics for Education Equity

Objective

To evaluate and compare a logistic regression model for identifying students at risk of dropping out based on early-term academic performance and behavioral indicators. This builds on the Random Forest model developed earlier, with an emphasis on interpretability, sparsity, and recall for early intervention.

Data Source

We used the same dataset as before:

UCI Student Performance Dataset

Combined student-mat.csv and student-por.csv into one dataset of 1,044 Portuguese secondary students

Includes demographic, academic, and aspirational features such as:

G1, G2, G3 (first-, second-, and final-period grades)

absences, studytime, failures

higher, romantic, paid, and more

Link: UCI Machine Learning Repository

Target Variable

To define a dropout proxy, we used:

df["dropped_out"] = ((df["G3"] < 10) | (df["absences"] > 30)).astype(int)

A student is flagged as at-risk if:

Final grade G3 is below 10 (failing), or

They have more than 30 absences

While not a perfect substitute for real dropout data, it aligns with likely disengagement or academic failure.

Preventing Label Leakage

G3 (final grade) was part of the label and thus excluded from predictors

G2 was also excluded as it comes too late in the term

G1 (first-term grade) was retained, representing early academic signal

This ensured our model was trained on only early-term and static data, making it viable for proactive intervention.

Preprocessing Pipeline

Numeric features were standardized with StandardScaler

Categorical variables were one-hot encoded using OneHotEncoder(handle_unknown="ignore")

All transformations wrapped in a ColumnTransformer, then piped into LogisticRegression

Model Configuration

We selected Logistic Regression with L1 (Lasso) regularization for:

Feature sparsity: eliminates irrelevant predictors

Interpretability: signs and magnitudes of coefficients are intuitive

Comparability: benchmark against Random Forest in same pipeline

param_grid = {
    'clf__C': [1.0, 0.5, 0.1, 0.01],
    'clf__penalty': ['l1'],
}

Used liblinear solver and class_weight="balanced"

Best parameters found via 5-fold GridSearchCV

Threshold adjusted to 0.35 to increase recall

Final Model Results

Metric

Class 0 (not at risk)

Class 1 (at risk)

Precision

0.98

0.55

Recall

0.78

0.94

F1-Score

0.87

0.69

ROC-AUC

-

0.935

Confusion Matrix:

[[126  36]
 [  3  44]]

Only 3 false negatives (low-risk missed students)

36 false positives (flagged students not truly at risk)

Top 5 Coefficients (Non-zero, Ordered by Magnitude)

Feature

Coefficient

Interpretation

G1

–2.47

Higher first-term grades lower dropout risk

absences

+0.34

More absences increase dropout risk

romantic_no

–0.33

Not having a romantic relationship lowers risk

failures

+0.30

More past failures increase risk

studytime

+0.13

More study time correlates with higher risk*

* Counterintuitive but likely reflects students struggling despite effort

Comparison to Random Forest

Aspect

Logistic Regression

Random Forest

ROC-AUC

0.935

0.914

Recall (class 1)

0.94

0.77

Precision (class 1)

0.55

0.61

Interpretability

High (coefficients)

Medium (SHAP needed)

Sparsity

High

Low

"Logistic regression achieved stronger recall with fewer features, making it highly usable for flagging students for early review. Random Forest offers more precision but less transparency."

Key Takeaways

First-term grades (G1) are the single strongest predictor of dropout risk.

Absences, failures, and lack of aspirational indicators (e.g., higher_no) are important secondary features.

L1 regularization enabled interpretability and sparsity, which are beneficial for stakeholder communication.

The model delivers high recall (94%), which is vital for ensuring that at-risk students are not missed.

Next Steps

Evaluate ElasticNet and Ridge logistic models for performance tuning

Add calibration plots to assess probabilistic trustworthiness

Combine with Random Forest (ensemble or voting) if higher precision is desired

Use the model as part of a triage dashboard for AVID or similar student success programs

This model demonstrates how logistic regression—properly regularized and tuned—can produce interpretable, effective, and actionable insights in educational settings for dropout prevention.
