# credit_card

## Problem Overview

The dataset for credit card fraud detection is highly imbalanced, with frauds accounting for only 0.17% of all transactions. The goal is to identify fraud transactions accurately while minimizing false positives. Here's a breakdown of the process followed to develop the model.

### Features
- **V_ Features**: Principal components obtained through PCA. Interpretation is left to the machine.
- **Time**: The number of seconds elapsed between each transaction and the first transaction.
- **Amount**: The transaction amount, which could be used in "example-dependent cost-sensitive learning."
- **Class**: The target feature: `1` indicates fraud, and `0` indicates genuine transactions.

The recommended evaluation metric is **Area Under the Precision-Recall Curve (AUPRC)** because of the highly imbalanced data.

### Exploratory Data Analysis (EDA)
- **Time Feature Analysis**:
  - Fraudulent transactions show distinct "spikiness" when visualized across time, whereas non-fraud transactions exhibit a more predictable pattern.
  - **Mann-Whitney U Test**: Statistical test between fraud and non-fraud based on the time of day. A low p-value (`1.45e-14`) suggests a significant difference.
  
- **Time of Day Analysis**:
  - Converted the **Time** feature to "Time of Day" (0-23 hours format).
  - **Fraud** transactions peak at 2 AM, 11 AM, and 6 PM, while **Non-fraud** transactions are more prevalent from 8 AM to 9 AM.

## Data Preprocessing
- Removed the target variable (`Class`) and split the data into **training** and **testing** sets using **StratifiedShuffleSplit**.
- Applied **StratifiedKFold** for cross-validation to maintain class distribution in each fold.

## Model Selection

I explored multiple models to determine the best for fraud detection, eventually using **CatBoostClassifier**, **LGBMClassifier**, and **XGBClassifier**.

### CatBoostClassifier
- **Initial Parameters**:
    ```python
    CatBoostClassifier(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        verbose=0,
        eval_metric='AUC',
        class_weights=[1, 5],
        early_stopping_rounds=50, 
        subsample=0.8,
        colsample_bylevel=0.8,
        l2_leaf_reg=3
    )
    ```
- **Evaluation Metrics**:
    - **AUC**: 0.9866
    - **F1-Score**: 0.8409
    - **Precision**: 0.8830
    - **Recall**: 0.8046

### LGBMClassifier
- **Initial Parameters**:
    ```python
    LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        scale_pos_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=3,
        random_state=42
    )
    ```
- **Evaluation Metrics (before fine-tuning)**:
    - **AUC**: 0.8491
    - **F1-Score**: 0.4709
    - **Precision**: 0.3583
    - **Recall**: 0.6956

- **Fine-Tuned Parameters**:
    ```python
    LGBMClassifier(
        n_estimators=500,
        learning_rate=0.01,
        max_depth=4,
        scale_pos_weight=1,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_alpha=3,
        random_state=42
    )
    ```
- **Evaluation Metrics (after fine-tuning)**:
    - **AUC**: 0.9847
    - **F1-Score**: 0.8336
    - **Precision**: 0.9453
    - **Recall**: 0.7488

### XGBClassifier
- **Parameters**:
    ```python
    XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        scale_pos_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=3,
        random_state=42,
        eval_metric='auc'
    )
    ```
- **Evaluation Metrics**:
    - **AUC**: 0.9851
    - **F1-Score**: 0.8727
    - **Precision**: 0.9489
    - **Recall**: 0.8097

## Model Stacking
To improve the model's performance, I decided to stack **CatBoostClassifier**, **LGBMClassifier**, and **XGBClassifier** using **LogisticRegression** as the meta-model.

### Stacking Process
- The models are trained using **StratifiedKFold** with 5 splits.
- Predictions from each base model are stored as Out-Of-Fold (OOF) predictions.
- The stacked model (**LogisticRegression**) is trained on these OOF predictions.

#### Stacking Model Performance:
- **AUC (Stacked)**: 0.9805
- **F1-Score (Stacked)**: 0.8646
- **Precision (Stacked)**: 0.8830
- **Recall (Stacked)**: 0.8469

## Conclusion
The stacked model demonstrates excellent performance in distinguishing between fraud and genuine transactions:
- **AUC**: 98.05% - The model can effectively separate fraudulent transactions from legitimate ones.
- **F1-Score**: 86.46% - Balanced trade-off between precision and recall.
- **Precision**: 88.30% - Very few false positives.
- **Recall**: 84.69% - Captures approximately 85% of fraud cases.

The stacked model significantly improves the performance, especially in terms of recall, which is crucial for fraud detection tasks.
