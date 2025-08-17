#!/usr/bin/env python3
"""
Credit Card Fraud Detection - Machine Learning Project
Author: AI/ML Engineer Intern
Date: 2024

This script implements a comprehensive machine learning solution for detecting 
fraudulent credit card transactions with focus on minimizing false negatives.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import joblib
from datetime import datetime
import os

# Machine learning libraries
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    roc_curve, precision_recall_curve, auc
)
from sklearn.utils.class_weight import compute_class_weight

# XGBoost
import xgboost as xgb

# Handling class imbalance
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN

# Suppress warnings
warnings.filterwarnings('ignore')

def setup_environment():
    """Setup plotting style and environment."""
    plt.style.use('default')
    sns.set_palette("husl")
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 12
    print("✅ Environment setup completed!")

def load_data():
    """Load the fraud detection dataset."""
    print("📥 Loading dataset...")
    
    try:
        # Try to load from data folder first
        df = pd.read_csv('data/fraud_detection.csv')
        print("✅ Dataset loaded from data/fraud_detection.csv")
    except FileNotFoundError:
        try:
            # Try alternative names
            df = pd.read_csv('data/fraud.csv')
            print("✅ Dataset loaded from data/fraud.csv")
        except FileNotFoundError:
            print("⚠️  Dataset not found in data/ folder")
            print("📥 Please download the dataset from: https://www.kaggle.com/datasets/kartik2112/fraud-detection")
            print("📁 Place it in the data/ folder and update the filename below")
            
            # Create a sample dataset structure for demonstration
            print("\n🔧 Creating sample dataset structure for demonstration...")
            np.random.seed(42)
            n_samples = 10000
            
            # Create synthetic data similar to credit card fraud dataset
            df = pd.DataFrame({
                'step': np.random.randint(1, 100, n_samples),
                'type': np.random.choice(['CASH_IN', 'CASH_OUT', 'DEBIT', 'PAYMENT', 'TRANSFER'], n_samples),
                'amount': np.random.exponential(1000, n_samples),
                'nameOrig': [f'C{i:06d}' for i in range(n_samples)],
                'oldbalanceOrg': np.random.exponential(5000, n_samples),
                'newbalanceOrig': np.random.exponential(5000, n_samples),
                'nameDest': [f'C{i:06d}' for i in range(n_samples)],
                'oldbalanceDest': np.random.exponential(5000, n_samples),
                'newbalanceDest': np.random.exponential(5000, n_samples),
                'isFraud': np.random.choice([0, 1], n_samples, p=[0.95, 0.05]),
                'isFlaggedFraud': np.random.choice([0, 1], n_samples, p=[0.99, 0.01])
            })
            print("✅ Sample dataset created for demonstration")
    
    return df

def explore_data(df):
    """Perform exploratory data analysis."""
    print("\n🔍 Starting Exploratory Data Analysis...")
    print("=" * 60)
    
    # Basic information
    print(f"📊 Dataset Overview:")
    print(f"Shape: {df.shape}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Display first few rows
    print("\n🔍 First 5 rows:")
    print(df.head())
    
    # Data types and missing values
    print("\n📋 Data Types and Missing Values:")
    info_df = pd.DataFrame({
        'Data Type': df.dtypes,
        'Non-Null Count': df.count(),
        'Missing Values': df.isnull().sum(),
        'Missing %': (df.isnull().sum() / len(df) * 100).round(2)
    })
    print(info_df)
    
    # Target variable analysis
    if 'isFraud' in df.columns:
        print("\n🎯 Target Variable Analysis (isFraud):")
        fraud_counts = df['isFraud'].value_counts()
        fraud_percentage = (fraud_counts / len(df) * 100).round(2)
        
        print(f"Legitimate transactions: {fraud_counts[0]:,} ({fraud_percentage[0]}%)")
        print(f"Fraudulent transactions: {fraud_counts[1]:,} ({fraud_percentage[1]}%)")
        print(f"Class imbalance ratio: {fraud_counts[0]/fraud_counts[1]:.1f}:1")
        
        # Create visualizations
        create_target_visualizations(fraud_counts)
    
    return df

def create_target_visualizations(fraud_counts):
    """Create visualizations for target variable."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Pie chart
    ax1.pie(fraud_counts.values, labels=['Legitimate', 'Fraudulent'], 
            autopct='%1.1f%%', startangle=90, colors=['lightgreen', 'lightcoral'])
    ax1.set_title('Transaction Distribution', fontsize=14, fontweight='bold')
    
    # Bar chart
    bars = ax2.bar(['Legitimate', 'Fraudulent'], fraud_counts.values, 
                   color=['lightgreen', 'lightcoral'])
    ax2.set_title('Transaction Counts', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Count')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{int(height):,}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.show()

def preprocess_data(df):
    """Preprocess the data for machine learning."""
    print("\n🔧 Starting Data Preprocessing...")
    print("=" * 60)
    
    # Create a copy for preprocessing
    df_processed = df.copy()
    print(f"📊 Original dataset shape: {df_processed.shape}")
    
    # Handle missing values
    print("\n🧹 Data Cleaning...")
    missing_summary = df_processed.isnull().sum()
    missing_percentage = (missing_summary / len(df_processed) * 100).round(2)
    
    missing_df = pd.DataFrame({
        'Missing_Count': missing_summary,
        'Missing_Percentage': missing_percentage
    })
    missing_df = missing_df[missing_df['Missing_Count'] > 0]
    
    if len(missing_df) > 0:
        print("Missing values found:")
        print(missing_df)
        
        # Handle missing values
        for col in missing_df.index:
            if df_processed[col].dtype in ['int64', 'float64']:
                # For numerical columns, fill with median
                median_val = df_processed[col].median()
                df_processed[col].fillna(median_val, inplace=True)
                print(f"✅ Filled missing values in {col} with median: {median_val}")
            else:
                # For categorical columns, fill with mode
                mode_val = df_processed[col].mode()[0]
                df_processed[col].fillna(mode_val, inplace=True)
                print(f"✅ Filled missing values in {col} with mode: {mode_val}")
    else:
        print("✅ No missing values found!")
    
    # Check for duplicates
    duplicates = df_processed.duplicated().sum()
    print(f"\n🔍 Duplicate rows: {duplicates}")
    if duplicates > 0:
        df_processed.drop_duplicates(inplace=True)
        print(f"✅ Removed {duplicates} duplicate rows")
    
    # Feature engineering
    print("\n🔤 Feature Engineering...")
    if 'amount' in df_processed.columns and 'oldbalanceOrg' in df_processed.columns:
        # Balance change features
        df_processed['balance_change_orig'] = df_processed['newbalanceOrig'] - df_processed['oldbalanceOrg']
        df_processed['balance_change_dest'] = df_processed['newbalanceDest'] - df_processed['oldbalanceDest']
        
        # Amount to balance ratio
        df_processed['amount_to_balance_ratio'] = df_processed['amount'] / (df_processed['oldbalanceOrg'] + 1)
        
        print("✅ Created balance change and ratio features")
    
    # Encode categorical variables
    categorical_cols = df_processed.select_dtypes(include=['object']).columns
    label_encoders = {}
    
    for col in categorical_cols:
        if col != 'nameOrig' and col != 'nameDest':  # Skip ID columns
            le = LabelEncoder()
            df_processed[f'{col}_encoded'] = le.fit_transform(df_processed[col])
            label_encoders[col] = le
            print(f"✅ Encoded {col} with {df_processed[col].nunique()} unique values")
    
    # Remove original categorical columns (keep encoded versions)
    columns_to_drop = [col for col in categorical_cols if col != 'nameOrig' and col != 'nameDest']
    df_processed.drop(columns=columns_to_drop, inplace=True)
    
    # Remove ID columns as they don't add predictive value
    if 'nameOrig' in df_processed.columns:
        df_processed.drop('nameOrig', axis=1, inplace=True)
    if 'nameDest' in df_processed.columns:
        df_processed.drop('nameDest', axis=1, inplace=True)
    
    print(f"\n📊 Dataset shape after preprocessing: {df_processed.shape}")
    print(f"🔤 Features after preprocessing: {list(df_processed.columns)}")
    
    return df_processed, label_encoders

def handle_class_imbalance(df_processed):
    """Handle class imbalance using SMOTE."""
    print("\n⚖️ Handling Class Imbalance...")
    
    # Check current class distribution
    if 'isFraud' in df_processed.columns:
        target_counts = df_processed['isFraud'].value_counts()
        print(f"\n📊 Current class distribution:")
        print(f"Legitimate: {target_counts[0]:,} ({target_counts[0]/len(df_processed)*100:.2f}%)")
        print(f"Fraudulent: {target_counts[1]:,} ({target_counts[1]/len(df_processed)*100:.2f}%)")
        
        # Prepare features and target
        X = df_processed.drop('isFraud', axis=1)
        y = df_processed['isFraud']
        
        print(f"\n🔤 Features shape: {X.shape}")
        print(f"🎯 Target shape: {y.shape}")
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"\n📊 Training set: {X_train.shape[0]:,} samples")
        print(f"📊 Test set: {X_test.shape[0]:,} samples")
        
        # Check training set class distribution
        train_fraud_rate = y_train.mean() * 100
        print(f"\n🎯 Training set fraud rate: {train_fraud_rate:.2f}%")
        
        # Apply SMOTE to training data only
        print("\n🔄 Applying SMOTE to training data...")
        smote = SMOTE(random_state=42, k_neighbors=5)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
        
        print(f"📊 Training set after SMOTE: {X_train_balanced.shape[0]:,} samples")
        print(f"🎯 Balanced fraud rate: {y_train_balanced.mean()*100:.2f}%")
        
        # Scale the features
        print("\n📏 Scaling features...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_balanced)
        X_test_scaled = scaler.transform(X_test)
        
        print("✅ Feature scaling completed!")
        
        # Save the scaler for later use
        os.makedirs('models', exist_ok=True)
        joblib.dump(scaler, 'models/scaler.pkl')
        print("💾 Scaler saved to models/scaler.pkl")
        
        return X_train_scaled, X_test_scaled, y_train_balanced, y_test, scaler
    
    return None, None, None, None, None

def train_models(X_train_scaled, y_train_balanced):
    """Train multiple machine learning models."""
    print("\n🤖 Starting Model Development...")
    print("=" * 60)
    
    # Define models to train
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
        'XGBoost': xgb.XGBClassifier(random_state=42, eval_metric='logloss')
    }
    
    # Dictionary to store results
    results = {}
    trained_models = {}
    
    print(f"🚀 Training {len(models)} models...")
    print("-" * 40)
    
    # Train each model
    for name, model in models.items():
        print(f"\n🔧 Training {name}...")
        
        # Train the model
        start_time = datetime.now()
        model.fit(X_train_scaled, y_train_balanced)
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc_roc = roc_auc_score(y_test, y_pred_proba)
        
        # Store results
        results[name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc_roc': auc_roc,
            'training_time': training_time
        }
        
        trained_models[name] = model
        
        print(f"✅ {name} trained in {training_time:.2f} seconds")
        print(f"   Accuracy: {accuracy:.4f}")
        print(f"   Precision: {precision:.4f}")
        print(f"   Recall: {recall:.4f}")
        print(f"   F1-Score: {f1:.4f}")
        print(f"   AUC-ROC: {auc_roc:.4f}")
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train_scaled, y_train_balanced, 
                                   cv=5, scoring='f1')
        print(f"   CV F1-Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    print("\n🎉 All models trained successfully!")
    
    # Save all trained models
    for name, model in trained_models.items():
        filename = f'models/{name.lower().replace(" ", "_")}.pkl'
        joblib.dump(model, filename)
        print(f"💾 {name} saved to {filename}")
    
    return results, trained_models

def evaluate_models(results, trained_models, X_test_scaled, y_test):
    """Evaluate and compare model performance."""
    print("\n📊 Starting Model Evaluation...")
    print("=" * 60)
    
    # Create results comparison DataFrame
    results_df = pd.DataFrame(results).T
    results_df = results_df.round(4)
    
    print("📈 Model Performance Comparison:")
    print(results_df)
    
    # Find best model for each metric
    print("\n🏆 Best Model by Metric:")
    for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']:
        best_model = results_df[metric].idxmax()
        best_score = results_df.loc[best_model, metric]
        print(f"• {metric.replace('_', ' ').title()}: {best_model} ({best_score:.4f})")
    
    # Create visualizations
    create_evaluation_visualizations(results_df, trained_models, X_test_scaled, y_test)
    
    return results_df

def create_evaluation_visualizations(results_df, trained_models, X_test_scaled, y_test):
    """Create comprehensive evaluation visualizations."""
    print("\n📊 Creating Evaluation Visualizations...")
    
    # 1. Performance comparison bar chart
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.ravel()
    
    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc', 'training_time']
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC', 'Training Time (s)']
    
    for i, (metric, name) in enumerate(zip(metrics, metric_names)):
        if i < 6:  # Ensure we don't exceed subplot count
            bars = axes[i].bar(results_df.index, results_df[metric], 
                               color=sns.color_palette("husl", len(results_df)))
            axes[i].set_title(f'{name} Comparison', fontsize=14, fontweight='bold')
            axes[i].set_ylabel(name)
            axes[i].tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                axes[i].text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                            f'{height:.4f}', ha='center', va='bottom', fontweight='bold')
            
            axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 2. Confusion matrices for all models
    print("\n🔍 Confusion Matrices:")
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    axes = axes.ravel()
    
    for i, (name, model) in enumerate(trained_models.items()):
        if i < 3:  # Ensure we don't exceed subplot count
            y_pred = model.predict(X_test_scaled)
            cm = confusion_matrix(y_test, y_pred)
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                        xticklabels=['Legitimate', 'Fraudulent'],
                        yticklabels=['Legitimate', 'Fraudulent'])
            axes[i].set_title(f'{name} - Confusion Matrix', fontsize=14, fontweight='bold')
            axes[i].set_xlabel('Predicted')
            axes[i].set_ylabel('Actual')
            
            # Add performance metrics as text
            tn, fp, fn, tp = cm.ravel()
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            axes[i].text(0.5, -0.3, f'Precision: {precision:.3f}\nRecall: {recall:.3f}\nF1: {f1:.3f}',
                        ha='center', va='center', transform=axes[i].transAxes,
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    # 3. ROC Curves
    print("\n📈 ROC Curves:")
    
    plt.figure(figsize=(10, 8))
    
    for name, model in trained_models.items():
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        auc_score = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, label=f'{name} (AUC = {auc_score:.4f})', linewidth=2)
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier (AUC = 0.5)')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves Comparison', fontsize=16, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def main():
    """Main function to run the complete fraud detection pipeline."""
    print("🚀 Credit Card Fraud Detection - Machine Learning Project")
    print("=" * 70)
    print(f"📅 Analysis started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Setup environment
    setup_environment()
    
    # Load data
    df = load_data()
    
    # Explore data
    df = explore_data(df)
    
    # Preprocess data
    df_processed, label_encoders = preprocess_data(df)
    
    # Handle class imbalance
    X_train_scaled, X_test_scaled, y_train_balanced, y_test, scaler = handle_class_imbalance(df_processed)
    
    if X_train_scaled is not None:
        # Train models
        results, trained_models = train_models(X_train_scaled, y_train_balanced)
        
        # Evaluate models
        results_df = evaluate_models(results, trained_models, X_test_scaled, y_test)
        
        # Final results
        print("\n🏆 Final Results and Business Impact Analysis")
        print("=" * 60)
        
        # Determine the best overall model
        best_model_name = results_df['f1_score'].idxmax()
        best_model = trained_models[best_model_name]
        best_results = results_df.loc[best_model_name]
        
        print(f"\n🥇 Best Overall Model: {best_model_name}")
        print(f"📊 F1-Score: {best_results['f1_score']:.4f}")
        print(f"🎯 Recall: {best_results['recall']:.4f}")
        print(f"⚖️  Precision: {best_results['precision']:.4f}")
        print(f"📈 AUC-ROC: {best_results['auc_roc']:.4f}")
        
        # Save the best model
        best_model_filename = f'models/best_model_{best_model_name.lower().replace(" ", "_")}.pkl'
        joblib.dump(best_model, best_model_filename)
        print(f"\n💾 Best model saved to: {best_model_filename}")
        
        print("\n🎉 Project completed successfully!")
        print(f"📅 Analysis completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        print("❌ Failed to prepare data for training. Please check the dataset.")

if __name__ == "__main__":
    main() 