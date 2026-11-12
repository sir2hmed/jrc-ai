# Phase 2 — Machine Learning Foundations

> **Program:** Cooperative Training in Artificial Intelligence — JRC-AI
> **Folder:** `Phase 2/`
# Phase 2 — Machine Learning Foundations

> **Program:** Cooperative Training in Artificial Intelligence — JRC-AI
> **Folder:** `Phase 2/`

## Overview

Phase 2 covers the end-to-end classical machine learning workflow (`Data → Preprocessing → Exploration → Features → Training → Evaluation → Interpretation`), including data preparation, core algorithms, model evaluation, and a student performance classification project.

## Folder Contents

| File | Type | Description |
|---|---|---|
| [`task1.pdf`](./task1.pdf) | PDF Report | Task 1 — Data preparation and preprocessing (Weka AI & Python) |
| [`task2.ipynb`](./task2.ipynb) | Notebook | Task 2 — Classical machine learning algorithms |
| [`task3.ipynb`](./task3.ipynb) | Notebook | Task 3 — Model evaluation, metrics, and generalization |
| [`classification_models.ipynb`](./classification_models.ipynb) | Notebook | Student performance classification project |
| [`student_performance_dataset.csv`](./student_performance_dataset.csv) | Dataset | Dataset used for student performance classification |
| [`Machine Learning Models.pptx`](./Machine%20Learning%20Models.pptx) | Presentation | Summary slides on machine learning models |
| [`8ml.png`](./8ml.png) | Visual | Overview of the 8 classical ML algorithms |
| [`CM.png`](./CM.png) | Visual | Confusion matrix visualization |
| [`RM.png`](./RM.png) | Visual | Regression metrics visualization |

## Tasks & Topics Summary

| Task | Focus | Key Concepts |
|---|---|---|
| **Task 1** | Data Preparation | Data cleaning, missing values, categorical encoding, scaling, feature engineering, EDA |
| **Task 2** | Classical ML | Linear & Logistic Regression, Decision Tree, Random Forest, SVM, KNN, Naive Bayes, K-Means |
| **Task 3** | Model Evaluation | Train/Val/Test split, Accuracy, Precision, Recall, F1, ROC-AUC, MAE, MSE, R², overfitting & data leakage |
| **Project** | Classification | End-to-end student performance prediction using `student_performance_dataset.csv` |

---
[← Back to Main README](../README.md)
## Overview

Phase 2 covers the end-to-end classical machine learning workflow (`Data → Preprocessing → Exploration → Features → Training → Evaluation → Interpretation`), including data preparation, core algorithms, model evaluation, and a student performance classification project.

## Folder Contents

| File | Type | Description |
|---|---|---|
| [`task1.pdf`](./task1.pdf) | PDF Report | Task 1 — Data preparation and preprocessing (Weka AI & Python) |
| [`task2.ipynb`](./task2.ipynb) | Notebook | Task 2 — Classical machine learning algorithms |
| [`task3.ipynb`](./task3.ipynb) | Notebook | Task 3 — Model evaluation, metrics, and generalization |
| [`classification_models.ipynb`](./classification_models.ipynb) | Notebook | Student performance classification project |
| [`student_performance_dataset.csv`](./student_performance_dataset.csv) | Dataset | Dataset used for student performance classification |
| [`Machine Learning Models.pptx`](./Machine%20Learning%20Models.pptx) | Presentation | Summary slides on machine learning models |
| [`8ml.png`](./8ml.png) | Visual | Overview of the 8 classical ML algorithms |
| [`CM.png`](./CM.png) | Visual | Confusion matrix visualization |
| [`RM.png`](./RM.png) | Visual | Regression metrics visualization |

## Tasks & Topics Summary

| Task | Focus | Key Concepts |
|---|---|---|
| **Task 1** | Data Preparation | Data cleaning, missing values, categorical encoding, scaling, feature engineering, EDA |
| **Task 2** | Classical ML | Linear & Logistic Regression, Decision Tree, Random Forest, SVM, KNN, Naive Bayes, K-Means |
| **Task 3** | Model Evaluation | Train/Val/Test split, Accuracy, Precision, Recall, F1, ROC-AUC, MAE, MSE, R², overfitting & data leakage |
| **Project** | Classification | End-to-end student performance prediction using `student_performance_dataset.csv` |

---
[← Back to Main README](../README.md) algorithms, model evaluation, and a student performance classification project.

## Phase 2 Tasks

### Task 1 – Data Preparation
Data preprocessing was performed using **Weka AI** and **Python**.

Main steps:
- Replace missing values
- Normalize numerical data to a 0–1 range
- Remove duplicate records
- Convert categorical/string values into numerical values
- Visualize the prepared data
- Apply a Decision Tree classifier to a small customer purchase dataset

### Task 2 – Machine Learning Algorithms
This task introduces the main machine learning categories and algorithms.

Covered topics:
- Regression
- Classification
- Clustering

Algorithms explained:
1. Linear Regression
2. Logistic Regression
3. Decision Tree
4. Random Forest
5. Support Vector Machine (SVM)
6. K-Nearest Neighbors (KNN)
7. Naive Bayes
8. K-Means

Simple examples using **scikit-learn** are included for regression, classification, and clustering.

### Task 3 – Model Evaluation
This task focuses on evaluating machine learning models.

Covered topics:
- Training, validation, and testing datasets
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- Classification Report
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- R² Score
- Overfitting and model generalization

## Phase 2 Project – Student Performance Classification

The project uses a dataset of **1,000 students** to predict the final grade:

`A, B, C, D, or F`

The data includes features such as:
- Study time
- Attendance
- Sleep hours
- Previous grade
- Final exam score
- Internet access
- Extracurricular activities
- Part-time job

Five classification models are trained and compared:

| Model | Accuracy |
|---|---:|
| Decision Tree | 99.5% |
| Random Forest | 99.0% |
| Logistic Regression | 94.5% |
| SVM | 88.0% |
| KNN | 77.0% |

The **Decision Tree** achieved the highest accuracy in the current experiment.

## Files

- `task1.pdf` – Data preparation task
- `task2.ipynb` – Machine learning algorithms
- `task3.ipynb` – Model evaluation
- `classification_models.ipynb` – Student performance classification notebook
- `student_performance_dataset.csv` – Project dataset
- `project.py` – Python version of the classification project

## Technologies

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook
- Weka AI
