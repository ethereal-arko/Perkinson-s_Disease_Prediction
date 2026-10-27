# Parkinson's Disease Prediction Using SVM

## Overview

This project uses machine learning to classify voice measurement data into two categories: Healthy/Control and Parkinson's-positive. A Support Vector Machine (SVM) classifier with a linear kernel is used for prediction.

The project demonstrates an end-to-end introductory machine learning workflow, including data inspection, exploratory data analysis, preprocessing, model training, evaluation, visualization, and prediction on a new sample.

**Disclaimer:** This is an educational machine learning project, not a clinically validated diagnostic system. Its predictions must not be used for medical diagnosis or treatment decisions.

## Objectives

- Explore and understand the Parkinson's voice measurement dataset.
- Inspect missing values, duplicate records, and target class distribution.
- Prepare numerical features for machine learning.
- Train a Linear SVM classifier.
- Evaluate performance using multiple classification metrics.
- Visualize class-wise performance and the confusion matrix.
- Build a function for predicting the class of a new sample.

## Dataset

The project uses `parkinsons.csv`, containing 195 voice recordings and 24 columns.

- `name`: Recording identifier.
- Numerical features: 22 voice measurement features.
- `status`: Target label.

### Target labels

| Label | Meaning |
|---|---|
| 0 | Healthy/Control |
| 1 | Parkinson's-positive |

The dataset contains 147 positive records and 48 control records. This class imbalance is considered during evaluation.

The `name` column is excluded from model training because it identifies recordings rather than representing a voice measurement.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Jupyter Notebook / Google Colab

## Methodology

1. **Data loading:** Load the dataset using Pandas.
2. **Data inspection:** Examine dataset dimensions, data types, missing values, duplicates, and target distribution.
3. **Exploratory data analysis:** Compare selected voice measurements across target classes.
4. **Feature selection:** Exclude the recording identifier and target column from the input features.
5. **Data splitting:** Separate training and testing records while keeping records from the same participant within the same split.
6. **Feature scaling:** Apply `StandardScaler` through a Scikit-learn pipeline.
7. **Model training:** Train an SVM classifier with a linear kernel.
8. **Evaluation:** Measure accuracy, precision, recall, F1-score, and confusion matrix.
9. **Prediction:** Use the trained pipeline to classify a new sample.

## Model Performance

On the selected group-based holdout test set, the model achieved:

| Metric | Score |
|---|---:|
| Training accuracy | 92.00% |
| Test accuracy | 84.44% |
| Parkinson's-positive precision | 82.50% |
| Parkinson's-positive recall | 100.00% |
| Parkinson's-positive F1-score | 90.41% |
| Healthy/Control recall | 41.67% |

### Confusion Matrix

| Actual / Predicted | Healthy/Control | Parkinson's-positive |
|---|---:|---:|
| Healthy/Control | 5 | 7 |
| Parkinson's-positive | 0 | 33 |

The model correctly classified all 33 positive test records, but it incorrectly classified 7 of the 12 control records as positive. Therefore, the overall accuracy alone does not fully describe the model's performance.

**Evaluation limitation:** The test set contains only 45 records from seven participants. These results are preliminary and may vary substantially with a different participant-level split. They should not be interpreted as an estimate of clinical diagnostic performance.

## Limitations and Future Improvements

- The dataset is relatively small and imbalanced.
- Healthy/Control recall is low in the current evaluation.
- The test set contains records from only seven participants.
- The current results come from a single group-based holdout split.
- The model has not been validated on an independent external dataset.
- Only a linear SVM has been evaluated in this experiment.

Possible improvements include repeated group-based evaluation, group-aware cross-validation, hyperparameter tuning, comparison with other classifiers, and testing on independent data.


