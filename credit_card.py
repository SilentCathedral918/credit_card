#!/usr/bin/env python
# coding: utf-8

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold, HalvingGridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, roc_curve

from xgboost import XGBClassifier
from catboost import CatBoostClassifier, Pool
from lightgbm import LGBMClassifier
from scipy.stats import mannwhitneyu

data = pd.read_csv("data/creditcard.csv", delimiter=',')
data['Time of Day'] = data['Time'] % 86400 // 3600

strat_split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in strat_split.split(data, data["Class"]):
    strat_train_set = data.loc[train_index]
    strat_test_set = data.loc[test_index]

X = strat_train_set.drop("Class", axis=1)
y = strat_train_set["Class"].copy()

y_test = strat_test_set["Class"].values
X_test = strat_test_set.drop("Class", axis=1)

# 3 models we would be perform stacking on: CatBoostClassifier, LGBMClassifier, and XGBClassifier
catboost = CatBoostClassifier(
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
lightgbm = LGBMClassifier(
    n_estimators=500,
    learning_rate=0.01,
    max_depth=4,
    scale_pos_weight=1,
    subsample=0.8,
    colsample_bytree=0.7,
    reg_alpha=3,
    random_state=42,
    verbose=-1
)
xgboost = XGBClassifier(
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

# model to be used as the one doing the stacking: LogisticRegression
stacker = LogisticRegression()

# prepare Out-Of-Fold predictions
n_splits = 5
stacked_skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
oof_preds = np.zeros((len(X), 3))  # 3 predictions for 3 base models
test_preds = np.zeros((len(strat_test_set), 3))

for fold, (train_idx, val_idx) in enumerate(stacked_skf.split(X, y)):
    # for every fold...
    print(f"Training fold {fold + 1}/{n_splits}...")
    
    # split the fold into train and validation subsets
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # train separately with each base model
    catboost.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=0)
    oof_preds[val_idx, 0] = catboost.predict_proba(X_val)[:, 1]
    
    lightgbm.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric='auc')
    oof_preds[val_idx, 1] = lightgbm.predict_proba(X_val)[:, 1]
    
    xgboost.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    oof_preds[val_idx, 2] = xgboost.predict_proba(X_val)[:, 1]

    # collect the test predictions for final evaluation
    test_preds[:, 0] += catboost.predict_proba(X_test)[:, 1]
    test_preds[:, 1] += lightgbm.predict_proba(X_test)[:, 1]
    test_preds[:, 2] += xgboost.predict_proba(X_test)[:, 1]

    print(f"Fold {fold + 1}/{n_splits} finished training.")

# when we are done with all folds, we train the stacker model using Out-Of-Fold predictions from the base models
stacker.fit(oof_preds, y)

test_preds /= n_splits
stacked_preds = stacker.predict_proba(test_preds)[:, 1]

# binarize predictions with threshold of 0.05 to further emphasize on detecting positive class
stacked_preds_bin = (stacked_preds >= 0.05).astype(int)

# calculate AUC, f1, precision, recall scores
auc = roc_auc_score(y_test, stacked_preds)
f1 = f1_score(y_test, stacked_preds_bin)
precision = precision_score(y_test, stacked_preds_bin)
recall = recall_score(y_test, stacked_preds_bin)

print(f"AUC (Stacked): {auc:.4f}")
print(f"Precision (Stacked): {precision:.4f}")
print(f"F1-score (Stacked): {f1:.4f}")
print(f"Recall (Stacked): {recall:.4f}")




