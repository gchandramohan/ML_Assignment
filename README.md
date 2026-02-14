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

## Model Comparison Table

  -----------------------------------------------------------------------------
  ML Model Name    Accuracy   AUC      Precision   Recall   F1 Score   MCC
  ---------------- ---------- -------- ----------- -------- ---------- --------
  Logistic         0.2787     0.6345   0.3577      0.2787   0.1912     0.1889
  Regression                                                           

  Decision Tree    0.3661     0.5611   0.2505      0.3661   0.2616     0.2264

  kNN              0.5439     0.7589   0.4281      0.5439   0.4554     0.5082

  Naive Bayes      0.3610     0.5684   0.1542      0.3610   0.2161     0.2239

  Random Forest    0.2666     0.7289   0.2173      0.2666   0.1201     0.0685
  (Ensemble)                                                           

  XGBoost          0.3492     0.7935   0.3113      0.3492   0.2475     0.2477
  (Ensemble)                                                           
  -----------------------------------------------------------------------------

------------------------------------------------------------------------

## Observations on Model Performance

  -----------------------------------------------------------------------
  ML Model Name        Observation about model performance
  -------------------- --------------------------------------------------
  Logistic Regression  Performs poorly due to non-linear class boundaries
                       in the dataset.

  Decision Tree        Better than Logistic Regression but may suffer
                       from overfitting.

  kNN                  Achieved the highest Accuracy and MCC. Performs
                       well due to similarity-based classification.

  Naive Bayes          Lower precision and F1 score due to strong
                       independence assumption between features.

  Random Forest        Moderate AUC but relatively lower classification
  (Ensemble)           accuracy in this configuration.

  XGBoost (Ensemble)   Achieved the highest AUC score, indicating strong
                       probabilistic ranking capability.
  -----------------------------------------------------------------------
