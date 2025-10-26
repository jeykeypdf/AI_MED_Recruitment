# Classical Machine Learning Methods

Below are short descriptions of classical machine learning methods that can be used to solve the classification task. For a deeper understanding, please refer to the official scikit-learn documentation or conduct independent research.

## Decision Trees

A method based on making decisions through a series of questions (data splits) in a tree-like structure. Each node represents a test on a selected feature, and each branch corresponds to the result of that test. Decision trees are easy to understand and interpret but are prone to overfitting. <br/>
📖 Documentation: [DecisionTreeClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)

## Random Forest

An ensemble algorithm that builds multiple decision trees on random subsets of data and features. The classification result is determined by majority voting. Random forests are more resistant to overfitting than single trees and provide strong predictive performance. <br/>
📖 Documentation: [RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)

## K-Nearest Neighbors (k-NN)

A similarity-based algorithm. A new sample is classified based on the classes of its `k` nearest points in the feature space (measured, for example, by Euclidean distance). The method is simple and effective but can be computationally expensive for large datasets. <br/>
📖 Documentation: [KNeighborsClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)

## Support Vector Machine (SVM)

A classification method that finds a hyperplane that best separates classes in the feature space. SVM aims to maximize the margin between classes, which provides good generalization. By using kernel functions, it can handle nonlinear classification problems. <br/>
📖 Documentation: [SVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)

## Logistic Regression

A probabilistic model used for binary classification. It estimates the probability of a sample belonging to a given class and classifies it based on a threshold (e.g., 0.5). It is a simple and interpretable model, often used as a baseline for classification tasks. A precursor to neural networks. <br/>
📖 Documentation: [LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)

# Classification Quality Metrics

To evaluate the performance of classification models, various metrics are used. The most common ones include:

## Accuracy

The proportion of correct predictions (both positive and negative) relative to all samples. It performs well on balanced datasets but can be misleading with imbalanced classes. <br/>
📖 Documentation: [accuracy_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html)

## Precision

Indicates what proportion of samples predicted as positive actually belong to the positive class. High precision means a low number of false positives. <br/>
📖 Documentation: [precision_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html)

## Recall

Shows what proportion of actual positive cases were correctly detected by the model. High recall means few false negatives. <br/>
📖 Documentation: [recall_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html)

## F1-score

The harmonic mean of precision and recall. The F1 value balances both metrics and is especially useful when dealing with imbalanced classes. <br/>
📖 Documentation: [f1_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html)

# Confusion Matrix

A confusion matrix visualizes the results of classification, distinguishing between correct and incorrect model predictions.

|                        | Actual Positive     | Actual Negative     |
| ---------------------- | ------------------- | ------------------- |
| **Predicted Positive** | True Positive (TP)  | False Positive (FP) |
| **Predicted Negative** | False Negative (FN) | True Negative (TN)  |

* **TP (True Positive)** – correctly identified positive cases.
* **TN (True Negative)** – correctly identified negative cases.
* **FP (False Positive)** – negative cases incorrectly classified as positive.
* **FN (False Negative)** – positive cases incorrectly classified as negative.

Metrics are calculated based on the values in this table:

* Accuracy = (TP + TN) / (TP + TN + FP + FN)
* Precision = TP / (TP + FP)
* Recall = TP / (TP + FN)
* F1-score = 2 * (Precision * Recall) / (Precision + Recall)

  <br/>

📖 Documentation: [confusion_matrix](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html)
