![](UTA-DataScience-Logo.png)

# Santander Customer Satisfaction Prediction

* **One Sentence Summary**: This repository contains an end-to-end machine learning pipeline to predict customer dissatisfaction using the Santander Customer Satisfaction dataset from [Kaggle](https://www.kaggle.com/competitions/santander-customer-satisfaction/overview).

---

## Overview

- **Task Definition**: The goal is to predict whether a customer is dissatisfied (`TARGET = 1`) based on anonymized numerical data. Success is evaluated using the **AUC** (Area Under the ROC Curve) metric.
- **Approach**: 
  - The problem is formulated as a binary classification task.
  - Data preprocessing includes low-variance feature removal and feature scaling.
  - Gradient Boosting Classifier was chosen as a baseline model for its efficiency in handling numerical features.
- **Performance Summary**: The Gradient Boosting Classifier achieved a validation AUC of **0.84**, indicating strong predictive power for distinguishing dissatisfied customers.

---

## Summary of Work Done

### Data

- **Input**:
  - Numerical features (anonymized) from a CSV file.
  - Train dataset: Includes a binary `TARGET` variable.
  - Test dataset: Includes only features (predictions submitted for evaluation).
- **Size**:
  - Training data: **76,020 rows × 371 columns**.
  - Test data: **75,028 rows × 370 columns**.

---

### Preprocessing / Cleanup

- Removed features with low variance (< 0.01) using the `VarianceThreshold` method.
- Applied standardization (`StandardScaler`) to scale numerical features.

---

### Data Visualization

- Plotted histograms comparing feature distributions for satisfied (`TARGET = 0`) and dissatisfied (`TARGET = 1`) customers.
- **Example Insights**:
  - Certain features show significant separation between the two classes, indicating their predictive potential.

---

### Problem Formulation

- **Input/Output**:
  - **Input**: Scaled numerical features.
  - **Output**: Predicted probabilities for customer dissatisfaction (`TARGET = 1`).
- **Model**: Gradient Boosting Classifier selected for its robust performance on numerical data.
- **Loss/Metric**: AUC used for evaluation, as specified in the Kaggle competition.

---

### Training

- **Setup**:
  - Split the training data into **80% training** and **20% validation** sets using stratified sampling.
  - Handled class imbalance through stratification during splitting.
- **Training**:
  - The Gradient Boosting Classifier was trained with default parameters for the baseline.
  - Training and evaluation completed in under 5 minutes on a local machine.

---

### Performance Comparison

| **Metric**       | **Validation Set** | 
|-------------------|--------------------|
| AUC (Validation) | 0.84               |

- The validation AUC demonstrates strong predictive power for customer dissatisfaction.

---

### Conclusions

- The Gradient Boosting Classifier proved to be an effective baseline model with minimal tuning.
- Feature selection and scaling were crucial to achieving strong model performance.

---

### Future Work

- Experiment with advanced models like **LightGBM** or **XGBoost** for improved results.
- Perform hyperparameter tuning to maximize AUC.
- Explore additional feature engineering techniques to improve separability between classes.

---

## How to Reproduce Results

### Software Setup

- Required Python packages:
  - `pandas`, `numpy`, `matplotlib`, `scikit-learn`.
- Install the packages:
  ```bash
  pip install pandas numpy matplotlib scikit-learn
