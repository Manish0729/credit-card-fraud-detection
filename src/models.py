"""
Model training and evaluation functions for Credit Card Fraud Detection project.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    roc_curve, precision_recall_curve, auc
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import joblib
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def get_models():
    """
    Get dictionary of models to train.
    
    Returns:
        dict: Dictionary of model names and instances
    """
    models = {
        'Logistic Regression': LogisticRegression(
            random_state=42, 
            max_iter=1000,
            class_weight='balanced'
        ),
        'Random Forest': RandomForestClassifier(
            random_state=42, 
            n_estimators=100,
            class_weight='balanced',
            max_depth=10
        ),
        'XGBoost': xgb.XGBClassifier(
            random_state=42, 
            eval_metric='logloss',
            scale_pos_weight=19,  # Approximate class imbalance ratio
            max_depth=6,
            learning_rate=0.1
        )
    }
    return models

def train_single_model(model, X_train, y_train, model_name):
    """
    Train a single model and return training time.
    
    Args:
        model: Model instance to train
        X_train: Training features
        y_train: Training target
        model_name: Name of the model
        
    Returns:
        tuple: (trained_model, training_time)
    """
    print(f"🔧 Training {model_name}...")
    
    start_time = datetime.now()
    model.fit(X_train, y_train)
    training_time = (datetime.now() - start_time).total_seconds()
    
    print(f"✅ {model_name} trained in {training_time:.2f} seconds")
    return model, training_time

def evaluate_single_model(model, X_test, y_test, model_name):
    """
    Evaluate a single model and return metrics.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test target
        model_name: Name of the model
        
    Returns:
        dict: Dictionary of evaluation metrics
    """
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc_roc = roc_auc_score(y_test, y_pred_proba)
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_test, y_test, cv=5, scoring='f1')
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc_roc': auc_roc,
        'cv_f1_mean': cv_scores.mean(),
        'cv_f1_std': cv_scores.std()
    }
    
    print(f"   📊 {model_name} Performance:")
    print(f"      Accuracy: {accuracy:.4f}")
    print(f"      Precision: {precision:.4f}")
    print(f"      Recall: {recall:.4f}")
    print(f"      F1-Score: {f1:.4f}")
    print(f"      AUC-ROC: {auc_roc:.4f}")
    print(f"      CV F1-Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    return metrics

def train_all_models(X_train, y_train, X_test, y_test):
    """
    Train and evaluate all models.
    
    Args:
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target
        
    Returns:
        tuple: (results_dict, trained_models_dict)
    """
    print("🤖 Starting Model Development...")
    print("=" * 60)
    
    models = get_models()
    results = {}
    trained_models = {}
    
    print(f"🚀 Training {len(models)} models...")
    print("-" * 40)
    
    for name, model in models.items():
        # Train model
        trained_model, training_time = train_single_model(model, X_train, y_train, name)
        trained_models[name] = trained_model
        
        # Evaluate model
        metrics = evaluate_single_model(trained_model, X_test, y_test, name)
        metrics['training_time'] = training_time
        results[name] = metrics
        
        print()  # Empty line for readability
    
    print("🎉 All models trained successfully!")
    return results, trained_models

def save_models(trained_models, scaler, label_encoders, base_path='models'):
    """
    Save all trained models and preprocessing objects.
    
    Args:
        trained_models: Dictionary of trained models
        scaler: Fitted scaler
        label_encoders: Dictionary of label encoders
        base_path: Base path to save models
    """
    print("💾 Saving models and preprocessing objects...")
    
    # Create models directory if it doesn't exist
    os.makedirs(base_path, exist_ok=True)
    
    # Save individual models
    for name, model in trained_models.items():
        filename = f'{base_path}/{name.lower().replace(" ", "_")}.pkl'
        joblib.dump(model, filename)
        print(f"✅ {name} saved to {filename}")
    
    # Save scaler
    scaler_filename = f'{base_path}/scaler.pkl'
    joblib.dump(scaler, scaler_filename)
    print(f"✅ Scaler saved to {scaler_filename}")
    
    # Save label encoders
    if label_encoders:
        encoders_filename = f'{base_path}/label_encoders.pkl'
        joblib.dump(label_encoders, encoders_filename)
        print(f"✅ Label encoders saved to {encoders_filename}")
    
    print("💾 All models and objects saved successfully!")

def load_models(base_path='models'):
    """
    Load all saved models and preprocessing objects.
    
    Args:
        base_path: Base path where models are saved
        
    Returns:
        tuple: (trained_models, scaler, label_encoders)
    """
    print("📥 Loading saved models and preprocessing objects...")
    
    trained_models = {}
    scaler = None
    label_encoders = None
    
    # Load individual models
    model_files = {
        'Logistic Regression': 'logistic_regression.pkl',
        'Random Forest': 'random_forest.pkl',
        'XGBoost': 'xgboost.pkl'
    }
    
    for name, filename in model_files.items():
        filepath = f'{base_path}/{filename}'
        if os.path.exists(filepath):
            trained_models[name] = joblib.load(filepath)
            print(f"✅ {name} loaded from {filepath}")
        else:
            print(f"⚠️  {filename} not found")
    
    # Load scaler
    scaler_filepath = f'{base_path}/scaler.pkl'
    if os.path.exists(scaler_filepath):
        scaler = joblib.load(scaler_filepath)
        print(f"✅ Scaler loaded from {scaler_filepath}")
    
    # Load label encoders
    encoders_filepath = f'{base_path}/label_encoders.pkl'
    if os.path.exists(encoders_filepath):
        label_encoders = joblib.load(encoders_filepath)
        print(f"✅ Label encoders loaded from {encoders_filepath}")
    
    return trained_models, scaler, label_encoders

def create_results_dataframe(results):
    """
    Create a formatted results DataFrame.
    
    Args:
        results: Dictionary of model results
        
    Returns:
        pd.DataFrame: Formatted results
    """
    results_df = pd.DataFrame(results).T
    results_df = results_df.round(4)
    
    # Reorder columns for better readability
    column_order = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc', 
                   'cv_f1_mean', 'cv_f1_std', 'training_time']
    results_df = results_df[column_order]
    
    return results_df

def print_model_summary(results_df):
    """
    Print comprehensive model performance summary.
    
    Args:
        results_df: Results DataFrame
    """
    print("🏆 Model Performance Summary")
    print("=" * 60)
    
    # Display results table
    print("\n📊 Performance Metrics:")
    print(results_df)
    
    # Find best model for each metric
    print("\n🏆 Best Model by Metric:")
    for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']:
        if metric in results_df.columns:
            best_model = results_df[metric].idxmax()
            best_score = results_df.loc[best_model, metric]
            print(f"• {metric.replace('_', ' ').title()}: {best_model} ({best_score:.4f})")
    
    # Overall best model (F1-score)
    if 'f1_score' in results_df.columns:
        best_overall = results_df['f1_score'].idxmax()
        best_f1 = results_df.loc[best_overall, 'f1_score']
        print(f"\n🥇 Best Overall Model: {best_overall} (F1-Score: {best_f1:.4f})")
    
    # Training time analysis
    if 'training_time' in results_df.columns:
        fastest_model = results_df['training_time'].idxmin()
        fastest_time = results_df.loc[fastest_model, 'training_time']
        print(f"⚡ Fastest Training: {fastest_model} ({fastest_time:.2f}s)")

def create_comprehensive_evaluation_plots(results_df, trained_models, X_test, y_test):
    """
    Create all evaluation visualizations.
    
    Args:
        results_df: Results DataFrame
        trained_models: Dictionary of trained models
        X_test: Test features
        y_test: Test target
    """
    print("📊 Creating comprehensive evaluation visualizations...")
    
    # 1. Performance comparison bar chart
    create_performance_comparison_plot(results_df)
    
    # 2. Confusion matrices
    create_confusion_matrices_plot(trained_models, X_test, y_test)
    
    # 3. ROC curves
    create_roc_curves_plot(trained_models, X_test, y_test)
    
    # 4. Precision-Recall curves
    create_precision_recall_curves_plot(trained_models, X_test, y_test)

def create_performance_comparison_plot(results_df):
    """Create performance comparison bar chart."""
    fig, axes = plt.subplots(2, 4, figsize=(20, 12))
    axes = axes.ravel()
    
    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc', 'cv_f1_mean', 'cv_f1_std', 'training_time']
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC', 'CV F1-Mean', 'CV F1-Std', 'Training Time (s)']
    
    for i, (metric, name) in enumerate(zip(metrics, metric_names)):
        if i < 8 and metric in results_df.columns:
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

def create_confusion_matrices_plot(trained_models, X_test, y_test):
    """Create confusion matrices for all models."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    axes = axes.ravel()
    
    for i, (name, model) in enumerate(trained_models.items()):
        if i < 3:
            y_pred = model.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                        xticklabels=['Legitimate', 'Fraudulent'],
                        yticklabels=['Legitimate', 'Fraudulent'])
            axes[i].set_title(f'{name} - Confusion Matrix', fontsize=14, fontweight='bold')
            axes[i].set_xlabel('Predicted')
            axes[i].set_ylabel('Actual')
            
            # Add performance metrics
            tn, fp, fn, tp = cm.ravel()
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            axes[i].text(0.5, -0.3, f'Precision: {precision:.3f}\nRecall: {recall:.3f}\nF1: {f1:.3f}',
                        ha='center', va='center', transform=axes[i].transAxes,
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.show()

def create_roc_curves_plot(trained_models, X_test, y_test):
    """Create ROC curves comparison."""
    plt.figure(figsize=(10, 8))
    
    for name, model in trained_models.items():
        y_pred_proba = model.predict_proba(X_test)[:, 1]
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

def create_precision_recall_curves_plot(trained_models, X_test, y_test):
    """Create Precision-Recall curves comparison."""
    plt.figure(figsize=(10, 8))
    
    for name, model in trained_models.items():
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        pr_auc = auc(recall, precision)
        
        plt.plot(recall, precision, label=f'{name} (AUC = {pr_auc:.4f})', linewidth=2)
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves Comparison', fontsize=16, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def get_best_model(results_df, trained_models, metric='f1_score'):
    """
    Get the best performing model based on specified metric.
    
    Args:
        results_df: Results DataFrame
        trained_models: Dictionary of trained models
        metric: Metric to use for selection
        
    Returns:
        tuple: (best_model_name, best_model_instance, best_score)
    """
    if metric not in results_df.columns:
        raise ValueError(f"Metric '{metric}' not found in results")
    
    best_model_name = results_df[metric].idxmax()
    best_model = trained_models[best_model_name]
    best_score = results_df.loc[best_model_name, metric]
    
    return best_model_name, best_model, best_score 