# Machine Learning Homework 4: SVM Classification of ICU Patient Survival

This repository contains the code and supplementary materials for the Machine Learning course (Tsinghua University Course 80250993) Homework 4. The task involves classifying patients' survival (0: survived; 1: dead) using 108 features from their Intensive Care Unit (ICU) records.

## Programming Environment

- **Operating System**: Windows 11
- **Python Version**: 3.10
- **Libraries and Versions**:
  - numpy: 1.19.2
  - pandas: 1.1.3
  - scikit-learn: 0.24.1
  - matplotlib: 3.3.2

## Dataset

- **Source**: Kaggle (WIDS Datathon 2020)
- **Dataset Name**: ICU Patient Dataset
- **Number of Features**: 108
- **Number of Samples**: Training Set - 5000, Test Set - 1097
- **Features**: A mixture of numeric and binary variables such as age, BMI, height, weight, heart rate, blood pressure, etc.

## Experiment Setup

The experiment involves training a Support Vector Machine (SVM) classifier using different kernel functions and parameter settings to classify patient survival. The following SVM configurations were used:

- **Kernel Functions**: Linear, Radial Basis Function (RBF), Sigmoid
- **Parameters**:
  - C values: 1 and 0.1 for Linear kernel
  - C value: 1 for RBF and Sigmoid kernels

## Files

- `train1_icu_data.csv`: Training set feature data.
- `train1_icu_label.csv`: Training set labels.
- `test1_icu_data.csv`: Test set feature data.
- `test1_icu_label.csv`: Test set labels.
- `svm_classification.ipynb`: Jupyter Notebook containing the SVM classification code.
- `experiment_report.pdf`: Detailed report of the experiment observations and analysis.
