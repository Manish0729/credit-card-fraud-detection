#!/usr/bin/env python3
"""
Fraud Detection System - Real Data Execution
This script runs the implemented fraud detection system on fraudTrain.csv,
prints clear metrics, and generates neutral, professional charts.
"""

import time
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import argparse

plt.style.use('default')
sns.set_palette('husl')


def clear_screen():
	os.system('clear' if os.name == 'posix' else 'cls')


def print_header():
	print("\n" + "=" * 100)
	print("Fraud Detection System - Real Data Execution")
	print("=" * 100)
	print("Dataset: fraudTrain.csv (1.3M+ transactions)")
	print("Models: XGBoost, Random Forest, Logistic Regression")
	print("Output: Metrics + Charts (ROC, Confusion, Detection Summary)")
	print("=" * 100)
	print()


def load_and_describe_dataset():
	print("Data Load & Overview")
	print("=" * 60)
	try:
		df = pd.read_csv('data/fraudTrain.csv')
		frauds = int(df['is_fraud'].sum())
		legits = int(len(df) - frauds)
		print(f"Rows: {len(df):,} | Fraud: {frauds:,} | Legitimate: {legits:,} | Fraud rate: {frauds/len(df)*100:.2f}%")
		print("Columns (first 10):", ", ".join(df.columns[:10]))
		print()
		return df
	except Exception as e:
		print("Error loading dataset:", e)
		return None


def preprocess_for_models(df_sample: pd.DataFrame):
	# Basic preprocessing to match trained features
	df = df_sample.copy()
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
	feature_cols = [
		'Unnamed: 0','amt','zip','lat','long','city_pop','merch_lat','merch_long',
		'amt_log','amt_squared','hour','day_of_week','month','category_encoded',
		'gender_encoded','state_encoded'
	]
	X = df[feature_cols]
	y = df['is_fraud']
	return X, y


def run_detection_and_charts(threshold: float = 0.5):
	print("Run Detection & Generate Charts")
	print("=" * 60)
	xgb = joblib.load('models/xgboost_real_sample.pkl')
	rf = joblib.load('models/random_forest_real_sample.pkl')
	lr = joblib.load('models/logistic_regression_real_sample.pkl')
	scaler = joblib.load('models/scaler_real_sample.pkl')
	
	df = pd.read_csv('data/fraudTrain.csv')
	df_sample = df.sample(n=min(100000, len(df)), random_state=42)
	X, y = preprocess_for_models(df_sample)
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
	X_test_s = scaler.transform(X_test)
	
	from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
	
	def eval_model(name, model, thr: float):
		proba = model.predict_proba(X_test_s)[:,1]
		pred = (proba >= thr).astype(int)
		acc = accuracy_score(y_test, pred)
		prec = precision_score(y_test, pred, zero_division=0)
		rec = recall_score(y_test, pred, zero_division=0)
		f1 = f1_score(y_test, pred, zero_division=0)
		auc = roc_auc_score(y_test, proba)
		print(f"{name:20s} | Thr: {thr:.2f} | Acc: {acc*100:5.1f}% | Prec: {prec*100:5.1f}% | Rec: {rec*100:5.1f}% | F1: {f1*100:5.1f}% | AUC: {auc*100:5.1f}%")
		return pred, proba
	
	print(f"Model Metrics (on real test split) using threshold={threshold:.2f}:")
	pred_lr, proba_lr = eval_model("Logistic Regression", lr, threshold)
	pred_rf, proba_rf = eval_model("Random Forest", rf, threshold)
	pred_xgb, proba_xgb = eval_model("XGBoost", xgb, threshold)
	
	# Charts (neutral filenames)
	os.makedirs('visualizations', exist_ok=True)
	
	# ROC curves
	plt.figure(figsize=(12,10))
	for name, proba in [
		("XGBoost", proba_xgb), ("Random Forest", proba_rf), ("Logistic Regression", proba_lr)
	]:
		fpr, tpr, _ = roc_curve(y_test, proba)
		plt.plot(fpr, tpr, linewidth=3, label=name)
	plt.plot([0,1],[0,1],'k--', alpha=0.5)
	plt.xlabel('False Positive Rate (FPR)'); plt.ylabel('True Positive Rate (TPR)')
	plt.title('ROC Curves - Real Data (fraudTrain.csv)')
	plt.legend(); plt.grid(True, alpha=0.3)
	plt.savefig('visualizations/report_roc_curves.png', dpi=300, bbox_inches='tight')
	plt.close()
	
	# Confusion matrices
	models = {"XGBoost": pred_xgb, "Random Forest": pred_rf, "Logistic Regression": pred_lr}
	fig, axes = plt.subplots(1,3, figsize=(20,6))
	for ax, (name, pred) in zip(axes, models.items()):
		cm = confusion_matrix(y_test, pred)
		sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
				xticklabels=['Legitimate','Fraudulent'], yticklabels=['Legitimate','Fraudulent'], ax=ax)
		ax.set_title(f"{name} @ Thr={threshold:.2f}"); ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
	plt.tight_layout(); plt.savefig('visualizations/report_confusion_matrices.png', dpi=300, bbox_inches='tight')
	plt.close()
	
	# Detection summary
	tp = ((y_test==1) & (pred_xgb==1)).sum()
	fn = ((y_test==1) & (pred_xgb==0)).sum()
	fp = ((y_test==0) & (pred_xgb==1)).sum()
	plt.figure(figsize=(8,6))
	cats = ['Frauds Detected','Frauds Missed','False Alarms']
	vals = [tp, fn, fp]
	colors = ['#2e7d32','#c62828','#ef6c00']
	bars = plt.bar(cats, vals, color=colors, alpha=0.85)
	for b,v in zip(bars, vals):
		plt.text(b.get_x()+b.get_width()/2, v+0.1, str(v), ha='center', va='bottom', fontweight='bold')
	plt.title(f'Detection Summary (XGBoost) - Real Data @ Thr={threshold:.2f}'); plt.ylabel('Count'); plt.grid(True, alpha=0.3)
	plt.savefig('visualizations/report_detection_summary.png', dpi=300, bbox_inches='tight')
	plt.close()
	
	print("\nCharts saved:")
	print("  visualizations/report_roc_curves.png")
	print("  visualizations/report_confusion_matrices.png")
	print("  visualizations/report_detection_summary.png")
	
	return True


def main():
	clear_screen(); print_header()
	print("This will run on real data and generate charts.\n")
	parser = argparse.ArgumentParser(description='Run fraud detection on real data.')
	parser.add_argument('--threshold', type=float, default=0.5, help='Decision threshold for converting probability to label (default: 0.5)')
	args = parser.parse_args()
	_ = input("Press Enter to start...")
	df = load_and_describe_dataset()
	if df is None:
		return
	run_detection_and_charts(threshold=args.threshold)
	print("\n" + "="*100)
	print("Run complete. You can now show the charts from the visualizations/ folder.")
	print("="*100)

if __name__ == '__main__':
	main()