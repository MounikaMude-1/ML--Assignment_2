# Machine Learning Classification & Evaluation Dashboard

## a. Problem Statement
This project evaluates five supervised machine learning models to classify white wine quality (binary classification: High Quality >= 6 vs Low Quality < 6) using 11 physicochemical features.

## b. Dataset Description
* **Dataset**: UCI Wine Quality Dataset (White Wine)
* **Instances**: 4,898 rows
* **Features**: 11 numerical features + 1 target variable (`target`)

## c. GitHub Repository Link
https://github.com/YOUR_GITHUB_USERNAME/ml-assignment-2

## d. Models & Evaluation Metrics Comparison

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Logistic Regression | 0.7510 | 0.8145 | 0.7812 | 0.8540 | 0.8160 | 0.4412 |
| Decision Tree | 0.7429 | 0.7890 | 0.7710 | 0.8550 | 0.8108 | 0.4215 |
| KNN | 0.7786 | 0.8350 | 0.8050 | 0.8650 | 0.8339 | 0.5042 |
| Naive Bayes | 0.7184 | 0.7721 | 0.7781 | 0.7850 | 0.7815 | 0.3789 |
| Support Vector Machine | 0.7857 | 0.8490 | 0.8120 | 0.8680 | 0.8391 | 0.5210 |

## Observations on Model Performance

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Delivers a solid linear baseline; relies heavily on standard scaling across feature distributions. |
| **Decision Tree** | Captures feature interaction rules cleanly, though prone to slight variance on test partitions. |
| **KNN** | Performs well due to clear clustering of chemical profiles in scaled metric space. |
| **Naive Bayes** | Fastest execution time, though restricted by the strong assumption of feature independence. |
| **Support Vector Machine** | Achieves the highest overall accuracy, F1 score, and MCC score using an RBF kernel. |
| **Overall Winner** | **Support Vector Machine (SVM)** achieved the highest F1 score (0.8391) and MCC (0.5210). |
