# Dry Bean Classification 

------------------------------------------------------------------------

## a) Problem Statement

The objective of this project is to classify different types of dry
beans based on their physical and morphological features using multiple
machine learning classification models. This is a multi-class
classification problem where each bean sample must be assigned to its
correct bean variety.

Six different classification models are implemented and compared using
multiple evaluation metrics to determine the best performing model.

------------------------------------------------------------------------

## b) Dataset Description

The dataset used is the **Dry Bean Dataset** obtained from Kaggle.

-   Total Instances: 13,611 samples
-   Total Features: 16 numerical features
-   Target Variable: Bean Class (Multi-class classification)
-   Classes: BARBUNYA, BOMBAY, CALI, DERMASON, HOROZ, SEKER, SIRA
-   All features are numerical measurements extracted from bean images.

------------------------------------------------------------------------

## c) Models Used & Comparison Table

The following six classification models were implemented:

1.  Logistic Regression
2.  Decision Tree
3.  K-Nearest Neighbors (KNN)
4.  Naive Bayes
5.  Random Forest (Ensemble)
6.  XGBoost (Ensemble)

------------------------------------------------------------------------

## Model Performance Comparison

| ML Model Name              | Accuracy |   AUC  | Precision | Recall | F1 Score |  MCC   |
|----------------------------|----------|--------|-----------|--------|----------|--------|
| Logistic Regression        | 0.2787   | 0.6345 | 0.3577    | 0.2787 | 0.1912   | 0.1889 |
| Decision Tree              | 0.4216   | 0.5927 | 0.2805    | 0.4216 | 0.3215   | 0.2933 |
| KNN                        | 0.5439   | 0.7589 | 0.4281    | 0.5439 | 0.4554   | 0.5082 |
| Naive Bayes                | 0.3610   | 0.5684 | 0.1542    | 0.3610 | 0.2161   | 0.2239 |
| Random Forest (Ensemble)   | 0.3639   | 0.8053 | 0.2120    | 0.3639 | 0.2361   | 0.2836 |
| XGBoost (Ensemble)         | 0.3621   | 0.7539 | 0.2467    | 0.3621 | 0.2426   | 0.2268 |


------------------------------------------------------------------------
## Model Observations

| ML Model Name            | Observation about Model Performance |
|--------------------------|--------------------------------------|
| Logistic Regression      | Performs poorly due to non-linear decision boundaries in the dataset, resulting in low overall accuracy and F1 score. |
| Decision Tree            | Shows improved performance compared to Logistic Regression with better F1 and MCC, but may still be prone to overfitting. |
| kNN                      | Achieved the highest Accuracy, F1 Score, and MCC among all models, indicating strong similarity-based classification performance. |
| Naive Bayes              | Lower precision and F1 score due to the strong independence assumption between features. |
| Random Forest (Ensemble) | Achieved the highest AUC score (0.8053), showing strong probabilistic ranking capability despite moderate accuracy. |
| XGBoost (Ensemble)       | Provides balanced performance with good AUC and stable classification results across classes. |


### Best Performing Model

Based on overall evaluation metrics, **KNN** is the best-performing model, achieving the highest Accuracy (0.5439), F1 Score (0.4554), and MCC (0.5082).

However, **Random Forest** achieved the highest AUC (0.8053), indicating strong class probability ranking capability.
