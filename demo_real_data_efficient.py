#!/usr/bin/env python3
"""
Efficient demo script for Credit Card Fraud Detection project with REAL dataset.
This version uses sampling to handle large datasets efficiently.
"""

import sys
import os

# Add src directory to path
sys.path.append('src')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve, auc
)
import xgboost as xgb
from imblearn.over_sampling import SMOTE
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def load_real_data_sample(sample_size=100000):
    """Load a sample of the real fraud dataset for efficient processing."""
    print("📥 Loading Real Fraud Dataset Sample...")
    
    try:
        # Load the full dataset
        df = pd.read_csv('data/fraudTrain.csv')
        print(f"✅ Full dataset loaded: {df.shape[0]:,} rows")
        
        # Check fraud distribution in full dataset
        if 'is_fraud' in df.columns:
            fraud_counts = df['is_fraud'].value_counts()
            print(f"📊 Full Dataset Fraud Distribution:")
            print(f"   • Legitimate: {fraud_counts[0]:,} ({fraud_counts[0]/len(df)*100:.2f}%)")
            print(f"   • Fraudulent: {fraud_counts[1]:,} ({fraud_counts[1]/len(df)*100:.2f}%)")
        
        # Take a stratified sample to maintain fraud ratio
        if 'is_fraud' in df.columns:
            # Stratified sampling to maintain fraud ratio
            fraud_sample = df[df['is_fraud'] == 1].sample(
                n=min(sample_size//10, len(df[df['is_fraud'] == 1])), 
                random_state=42
            )
            legit_sample = df[df['is_fraud'] == 0].sample(
                n=sample_size - len(fraud_sample), 
                random_state=42
            )
            df_sample = pd.concat([fraud_sample, legit_sample])
        else:
            # If no target column, just take random sample
            df_sample = df.sample(n=sample_size, random_state=42)
        
        print(f"📊 Sample dataset created: {df_sample.shape[0]:,} rows")
        print(f"💾 Sample memory usage: {df_sample.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        return df_sample
        
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return None

def explore_real_data(df):
    """Explore the real dataset structure."""
    print("\n🔍 Exploring Real Dataset...")
    print("=" * 50)
    
    # Display basic info
    print("📋 Dataset Info:")
    print(f"• Rows: {df.shape[0]:,}")
    print(f"• Columns: {df.shape[1]}")
    print(f"• Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Check for target variable
    if 'is_fraud' in df.columns:
        print("🎯 Target variable 'is_fraud' found!")
        
        # Analyze fraud distribution
        fraud_counts = df['is_fraud'].value_counts()
        fraud_percentage = (fraud_counts / len(df) * 100).round(2)
        
        print(f"\n📊 Fraud Distribution:")
        print(f"• Legitimate transactions: {fraud_counts[0]:,} ({fraud_percentage[0]}%)")
        print(f"• Fraudulent transactions: {fraud_counts[1]:,} ({fraud_percentage[1]}%)")
        print(f"• Class imbalance ratio: {fraud_counts[0]/fraud_counts[1]:.1f}:1")
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Pie chart
        colors = ['#2E8B57', '#DC143C']
        ax1.pie(fraud_counts.values, labels=['Legitimate', 'Fraudulent'], 
                autopct='%1.1f%%', startangle=90, colors=colors, explode=(0, 0.1))
        ax1.set_title('Real Dataset - Transaction Distribution', fontsize=16, fontweight='bold')
        
        # Bar chart
        bars = ax2.bar(['Legitimate', 'Fraudulent'], fraud_counts.values, color=colors)
        ax2.set_title('Real Dataset - Transaction Counts', fontsize=16, fontweight='bold')
        ax2.set_ylabel('Count')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{int(height):,}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.show()
        
    else:
        print("⚠️  Target variable 'is_fraud' not found!")
        print("Available columns:", list(df.columns))
    
    # Show sample data
    print("\n🔍 Sample Data (first 3 rows):")
    print(df.head(3))
    
    # Show column types
    print("\n📋 Column Types:")
    print(df.dtypes.value_counts())
    
    return df

def preprocess_real_data(df):
    """Preprocess the real dataset for machine learning."""
    print("\n🔧 Preprocessing Real Dataset...")
    print("=" * 50)
    
    df_clean = df.copy()
    
    # Remove the first column (index)
    if df_clean.columns[0] == '':
        df_clean = df_clean.iloc[:, 1:]
        print("✅ Removed index column")
    
    # Handle missing values
    missing_summary = df_clean.isnull().sum()
    missing_percentage = (missing_summary / len(df_clean) * 100).round(2)
    
    missing_df = pd.DataFrame({
        'Missing_Count': missing_summary,
        'Missing_Percentage': missing_percentage
    })
    missing_df = missing_df[missing_df['Missing_Count'] > 0]
    
    if len(missing_df) > 0:
        print("Missing values found:")
        print(missing_df)
        
        # Fill missing values
        for col in missing_df.index:
            if df_clean[col].dtype in ['int64', 'float64']:
                median_val = df_clean[col].median()
                df_clean[col].fillna(median_val, inplace=True)
                print(f"✅ Filled missing values in {col} with median: {median_val}")
            else:
                mode_val = df_clean[col].mode()[0]
                df_clean[col].fillna(mode_val, inplace=True)
                print(f"✅ Filled missing values in {col} with mode: {mode_val}")
    else:
        print("✅ No missing values found!")
    
    # Feature engineering
    print("\n🔤 Feature Engineering...")
    
    # Convert transaction amount to numeric
    if 'amt' in df_clean.columns:
        df_clean['amt'] = pd.to_numeric(df_clean['amt'], errors='coerce')
        df_clean['amt_log'] = np.log1p(df_clean['amt'])
        df_clean['amt_squared'] = df_clean['amt'] ** 2
        print("✅ Created amount features")
    
    # Extract time features and remove the original datetime column
    if 'trans_date_trans_time' in df_clean.columns:
        df_clean['trans_date_trans_time'] = pd.to_datetime(df_clean['trans_date_trans_time'])
        df_clean['hour'] = df_clean['trans_date_trans_time'].dt.hour
        df_clean['day_of_week'] = df_clean['trans_date_trans_time'].dt.dayofweek
        df_clean['month'] = df_clean['trans_date_trans_time'].dt.month
        # Remove the original datetime column to avoid dtype conflicts
        df_clean.drop(columns=['trans_date_trans_time'], inplace=True)
        print("✅ Created time features and removed datetime column")
    
    # Create dummy variables for categorical columns
    categorical_cols = ['category', 'gender', 'state']
    for col in categorical_cols:
        if col in df_clean.columns:
            dummies = pd.get_dummies(df_clean[col], prefix=col, drop_first=True)
            df_clean = pd.concat([df_clean, dummies], axis=1)
            print(f"✅ Created dummy variables for {col}")
    
    # Encode other categorical variables (limit to avoid too many features)
    label_encoders = {}
    categorical_cols = df_clean.select_dtypes(include=['object']).columns
    
    for col in categorical_cols:
        if df_clean[col].nunique() < 100:  # Limit to avoid too many features
            le = LabelEncoder()
            df_clean[f'{col}_encoded'] = le.fit_transform(df_clean[col])
            label_encoders[col] = le
            print(f"✅ Encoded {col}")
    
    # Remove original categorical columns
    df_clean.drop(columns=categorical_cols, inplace=True)
    
    # Ensure all columns are numeric
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    df_clean = df_clean[numeric_cols]
    
    print(f"\n📊 Dataset shape after preprocessing: {df_clean.shape}")
    print(f"🔢 All columns are now numeric: {df_clean.dtypes.unique()}")
    return df_clean, label_encoders

def prepare_features_and_target(df, target_col='is_fraud'):
    """Prepare features and target for machine learning."""
    print("\n🎯 Preparing Features and Target...")
    
    # Ensure target column exists
    if target_col not in df.columns:
        print(f"❌ Target column '{target_col}' not found!")
        print("Available columns:", list(df.columns))
        return None, None, None, None
    
    # Prepare features (exclude target and some ID columns)
    exclude_cols = [target_col, 'cc_num', 'trans_num', 'unix_time']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    X = df[feature_cols]
    y = df[target_col]
    
    print(f"🔤 Features shape: {X.shape}")
    print(f"🎯 Target shape: {y.shape}")
    print(f"📋 Feature columns: {len(feature_cols)}")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n📊 Data split:")
    print(f"   Training set: {X_train.shape[0]:,} samples")
    print(f"   Test set: {X_test.shape[0]:,} samples")
    
    return X_train, X_test, y_train, y_test

def handle_class_imbalance(X_train, y_train):
    """Handle class imbalance using SMOTE."""
    print("\n⚖️ Handling Class Imbalance...")
    
    # Check current distribution
    class_counts = y_train.value_counts()
    print(f"📊 Current class distribution:")
    for class_val, count in class_counts.items():
        percentage = (count / len(y_train) * 100)
        print(f"   Class {class_val}: {count:,} ({percentage:.2f}%)")
    
    # Apply SMOTE
    smote = SMOTE(random_state=42, k_neighbors=5)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    
    # Check balanced distribution
    balanced_counts = y_train_balanced.value_counts()
    print(f"📊 Balanced class distribution:")
    for class_val, count in balanced_counts.items():
        percentage = (count / len(y_train_balanced) * 100)
        print(f"   Class {class_val}: {count:,} ({percentage:.2f}%)")
    
    return X_train_balanced, y_train_balanced

def train_models(X_train, y_train, X_test, y_test):
    """Train multiple machine learning models."""
    print("\n🤖 Training Models...")
    print("=" * 50)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("✅ Features scaled")
    
    # Define models (using smaller Random Forest for efficiency)
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(random_state=42, n_estimators=50, max_depth=10),
        'XGBoost': xgb.XGBClassifier(random_state=42, eval_metric='logloss', n_estimators=100, max_depth=6)
    }
    
    results = {}
    trained_models = {}
    
    for name, model in models.items():
        print(f"\n🔧 Training {name}...")
        
        # Train model
        start_time = datetime.now()
        model.fit(X_train_scaled, y_train)
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
    
    return results, trained_models, scaler

def evaluate_models(results, trained_models, X_test, y_test):
    """Evaluate and compare model performance."""
    print("\n📊 Model Evaluation...")
    print("=" * 50)
    
    # Create results DataFrame
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
    
    # Overall best model (F1-score)
    best_overall = results_df['f1_score'].idxmax()
    best_f1 = results_df.loc[best_overall, 'f1_score']
    print(f"\n🥇 Best Overall Model: {best_overall} (F1-Score: {best_f1:.4f})")
    
    return results_df

def save_models(trained_models, scaler, label_encoders):
    """Save all trained models and preprocessing objects."""
    print("\n💾 Saving Models...")
    print("=" * 50)
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Save individual models
    for name, model in trained_models.items():
        filename = f'models/{name.lower().replace(" ", "_")}_real_sample.pkl'
        joblib.dump(model, filename)
        print(f"✅ {name} saved to {filename}")
    
    # Save scaler
    scaler_filename = 'models/scaler_real_sample.pkl'
    joblib.dump(scaler, scaler_filename)
    print(f"✅ Scaler saved to {scaler_filename}")
    
    # Save label encoders
    if label_encoders:
        encoders_filename = 'models/label_encoders_real_sample.pkl'
        joblib.dump(label_encoders, encoders_filename)
        print(f"✅ Label encoders saved to {encoders_filename}")
    
    print("💾 All models saved successfully!")

def main():
    """Main function to run the efficient fraud detection pipeline with real data."""
    print("🚀 Credit Card Fraud Detection - EFFICIENT REAL DATA Pipeline")
    print("=" * 70)
    print(f"📅 Analysis started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Step 1: Load real data sample
    df = load_real_data_sample(sample_size=100000)  # Use 100K sample for efficiency
    if df is None:
        return
    
    # Step 2: Explore data
    df = explore_real_data(df)
    
    # Step 3: Preprocess data
    df_processed, label_encoders = preprocess_real_data(df)
    
    # Step 4: Prepare features and target
    result = prepare_features_and_target(df_processed)
    if result is None:
        return
    
    X_train, X_test, y_train, y_test = result
    
    # Step 5: Handle class imbalance
    X_train_balanced, y_train_balanced = handle_class_imbalance(X_train, y_train)
    
    # Step 6: Train models
    results, trained_models, scaler = train_models(
        X_train_balanced, y_train_balanced, X_test, y_test
    )
    
    # Step 7: Evaluate models
    results_df = evaluate_models(results, trained_models, X_test, y_test)
    
    # Step 8: Save models
    save_models(trained_models, scaler, label_encoders)
    
    # Final summary
    print("\n🎉 EFFICIENT REAL DATA Pipeline Completed Successfully!")
    print("=" * 50)
    print(f"📊 Total models trained: {len(trained_models)}")
    print(f"🔤 Features used: {X_train.shape[1]}")
    print(f"📈 Training samples: {X_train_balanced.shape[0]:,}")
    print(f"🧪 Test samples: {X_test.shape[0]:,}")
    
    best_model = results_df['f1_score'].idxmax()
    best_f1 = results_df.loc[best_model, 'f1_score']
    print(f"🥇 Best performing model: {best_model}")
    print(f"📊 Best F1-Score: {best_f1:.4f}")
    
    print(f"\n📅 Analysis completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main() 