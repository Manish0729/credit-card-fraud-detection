#!/usr/bin/env python3
"""
Threshold Tuning on Real Data (fraudTrain.csv)
- Loads XGBoost model and scaler
- Evaluates precision/recall/F1 across thresholds
- Generates:
  * report_precision_recall_curve.png
  * report_threshold_tuning.png
  * report_confusion_best_threshold.png
- Prints recommended threshold that keeps high recall while improving precision
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import (
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

plt.style.use('default')
sns.set_palette('husl')

FEATURE_COLS = [
    'Unnamed: 0', 'amt', 'zip', 'lat', 'long', 'city_pop', 'merch_lat', 'merch_long',
    'amt_log', 'amt_squared', 'hour', 'day_of_week', 'month', 'category_encoded',
    'gender_encoded', 'state_encoded'
]


def preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    df = df.copy()
    df = df.fillna(0).drop_duplicates()
    df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
    df['hour'] = df['trans_date_trans_time'].dt.hour
    df['day_of_week'] = df['trans_date_trans_time'].dt.dayofweek
    df['month'] = df['trans_date_trans_time'].dt.month
    df['amt_log'] = np.log1p(df['amt'])
    df['amt_squared'] = df['amt'] ** 2
    le_gender = LabelEncoder(); le_category = LabelEncoder(); le_state = LabelEncoder()
    df['gender_encoded'] = le_gender.fit_transform(df['gender'])
    df['category_encoded'] = le_category.fit_transform(df['category'])
    df['state_encoded'] = le_state.fit_transform(df['state'])
    X = df[FEATURE_COLS]
    y = df['is_fraud']
    return X, y


def main():
    print("Loading model and data...")
    xgb = joblib.load('models/xgboost_real_sample.pkl')
    scaler = joblib.load('models/scaler_real_sample.pkl')

    df = pd.read_csv('data/fraudTrain.csv')
    df_sample = df.sample(n=min(100000, len(df)), random_state=42)
    X, y = preprocess(df_sample)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    X_test_s = scaler.transform(X_test)

    y_proba = xgb.predict_proba(X_test_s)[:, 1]

    # Precision-Recall curve
    precision, recall, pr_thresholds = precision_recall_curve(y_test, y_proba)

    os.makedirs('visualizations', exist_ok=True)
    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, linewidth=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve - Real Data (XGBoost)')
    plt.grid(True, alpha=0.3)
    plt.savefig('visualizations/report_precision_recall_curve.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Threshold sweep
    thresholds = np.linspace(0.05, 0.99, 40)
    precs, recs, f1s, fps, tps, fns = [], [], [], [], [], []

    for t in thresholds:
        y_pred_t = (y_proba >= t).astype(int)
        precs.append(precision_score(y_test, y_pred_t, zero_division=0))
        recs.append(recall_score(y_test, y_pred_t, zero_division=0))
        f1s.append(f1_score(y_test, y_pred_t, zero_division=0))
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred_t).ravel()
        fps.append(fp)
        tps.append(tp)
        fns.append(fn)

    # Plot threshold tuning
    plt.figure(figsize=(12, 8))
    plt.plot(thresholds, precs, label='Precision', linewidth=2)
    plt.plot(thresholds, recs, label='Recall', linewidth=2)
    plt.plot(thresholds, f1s, label='F1-Score', linewidth=2)
    plt.xlabel('Decision Threshold')
    plt.ylabel('Score')
    plt.title('Threshold Tuning - Precision/Recall/F1 vs Threshold (XGBoost)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('visualizations/report_threshold_tuning.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Choose recommended threshold: maximize F1 with constraint recall >= 0.95
    rec_thresh = 0.0
    best_f1 = -1.0
    for t, f1, r in zip(thresholds, f1s, recs):
        if r >= 0.95 and f1 > best_f1:
            best_f1 = f1
            rec_thresh = t

    if best_f1 < 0:  # no threshold meets recall >= 0.95, fall back to max F1
        rec_idx = int(np.argmax(f1s))
        rec_thresh = float(thresholds[rec_idx])

    y_pred_best = (y_proba >= rec_thresh).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_best).ravel()
    prec_best = precision_score(y_test, y_pred_best, zero_division=0)
    rec_best = recall_score(y_test, y_pred_best, zero_division=0)
    f1_best = f1_score(y_test, y_pred_best, zero_division=0)

    # Confusion matrix at best threshold
    plt.figure(figsize=(6, 5))
    sns.heatmap([[tn, fp], [fn, tp]], annot=True, fmt='d', cmap='Blues',
                xticklabels=['Legitimate', 'Fraudulent'],
                yticklabels=['Legitimate', 'Fraudulent'])
    plt.title(f'Confusion Matrix @ Threshold={rec_thresh:.2f}')
    plt.xlabel('Predicted'); plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig('visualizations/report_confusion_best_threshold.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("\nRecommended threshold (recall>=0.95 constraint): {:.2f}".format(rec_thresh))
    print("Metrics @ threshold:")
    print("  Precision: {:.3f}".format(prec_best))
    print("  Recall:    {:.3f}".format(rec_best))
    print("  F1-Score:  {:.3f}".format(f1_best))
    print("  TP: {} | FP: {} | FN: {} | TN: {}".format(tp, fp, fn, tn))
    print("\nSaved charts:")
    print("  visualizations/report_precision_recall_curve.png")
    print("  visualizations/report_threshold_tuning.png")
    print("  visualizations/report_confusion_best_threshold.png")


if __name__ == '__main__':
    main()