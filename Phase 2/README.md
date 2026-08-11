# Phase 2 – Machine Learning Tasks & Project

This repository contains my **Phase 2 machine learning tasks and project**, covering data preparation, machine learning algorithms, model evaluation, and a student performance classification project.

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
