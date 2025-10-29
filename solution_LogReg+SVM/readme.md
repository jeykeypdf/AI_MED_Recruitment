# Cardiomegaly Classification Task

This document summarizes the solution and findings from the `solution.ipynb` notebook.

---

## 1. Data Analysis and Preparation

On loading the `task_data.csv`, two major challenges were clear:

* **Small Dataset:** Only **37 samples** were available.
* **Class Imbalance:** The data was heavily skewed, with **~76% 'Sick' (Class 1)** and **~24% 'Healthy' (Class 0)**.

This small, imbalanced dataset meant that results from a single `train_test_split` would be unreliable.
Therefore, **Cross-Validation (CV)** was the most important tool for this task.

### Data Preprocessing

Two main preprocessing steps were applied:

1. **Fixing data types** for columns that used a comma as a decimal separator (e.g., *Heart perimeter*).
2. **Normalizing features** using `StandardScaler`, which is critical for the models used later.

---

## 2. Model Selection and Iteration

The process followed these stages:

### Baseline Model — Logistic Regression

* Started with a simple `LogisticRegression` model.
* As expected, performance was poor — heavily biased due to data imbalance.

**Result:**

* F1-Macro ≈ **0.62**
* Recall (Healthy) ≈ **0.40**

---

### Iteration 1 — Balanced Logistic Regression

To correct the bias, `LogisticRegression(class_weight='balanced')` was used.
This adjustment forces the model to pay more attention to the minority class.

**Result:**

* Recall (Healthy) ↑ **~0.70**
* F1-Macro ↑ **~0.64**

---

### Iteration 2 — Balanced Support Vector Machine (SVM)

To introduce more creativity, a `SVC(class_weight='balanced')` model was tested —
known for strong performance on small datasets.

**Result (Winner):**

* **Avg. F1-score (Macro): 0.7309**
* **Avg. Recall (Healthy): 0.7000**
* **Avg. Recall (Sick): 0.8200**

The **Balanced SVM** clearly outperformed all Logistic Regression variants.

---

## 3. Feature Engineering (Experiment)

A new feature was tested:
`Heart_Lung_Area_Ratio = Heart_Area / Lung_Area`

**Outcome:**

* Made Logistic Regression slightly worse.
* Had **no effect** on the SVC model (identical F1-score).

This indicates that the SVM already captured this relationship inherently.

---

## 4. Conclusion

The **final, best-performing model** for this task is:

> **Balanced Support Vector Machine (SVC(class_weight='balanced'))**,
> trained **without** the engineered feature.

**Final Results:**

* **F1-Macro:** 0.7309
* **Consistent and stable performance** across folds

This model was retrained on the full training set and evaluated on the test set to produce the final **classification report** and **ROC curves** shown at the end of the notebook.
