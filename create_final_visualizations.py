#!/usr/bin/env python3
"""
Create Professional Visualizations for Credit Card Fraud Detection Project
This script generates all the charts needed for portfolio presentation.
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, precision_recall_curve,
    classification_report
)
import warnings
warnings.filterwarnings('ignore')

# Set professional plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def setup_plotting_style():
    """Setup professional plotting style for portfolio."""
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 12
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['savefig.bbox'] = 'tight'
    
    # Set color palette
    colors = ['#2E8B57', '#DC143C', '#4169E1', '#FFD700', '#FF6347']
    sns.set_palette(colors)

def load_models_and_data():
    """Load the trained models and test data."""
    print("📥 Loading Models and Data...")
    
    try:
        # Load models
        xgb_model = joblib.load('models/xgboost_real_sample.pkl')
        rf_model = joblib.load('models/random_forest_real_sample.pkl')
        lr_model = joblib.load('models/logistic_regression_real_sample.pkl')
        scaler = joblib.load('models/scaler_real_sample.pkl')
        
        print("✅ Models loaded successfully!")
        
        # Load sample data for demonstration
        from demo_real_data_efficient import load_real_data_sample, preprocess_real_data, prepare_features_and_target
        
        df = load_real_data_sample(sample_size=50000)  # Smaller sample for visualization
        df_processed, _ = preprocess_real_data(df)
        X_train, X_test, y_train, y_test = prepare_features_and_target(df_processed)
        
        # Scale features
        X_test_scaled = scaler.transform(X_test)
        
        return {
            'XGBoost': xgb_model,
            'Random Forest': rf_model,
            'Logistic Regression': lr_model
        }, X_test_scaled, y_test
        
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        print("Creating sample data for demonstration...")
        return create_sample_models_and_data()

def create_sample_models_and_data():
    """Create sample models and data for demonstration if real models not available."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    import xgboost as xgb
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    
    # Create sample data
    X, y = make_classification(n_samples=10000, n_features=20, n_informative=15, 
                              n_redundant=5, n_clusters_per_class=1, 
                              weights=[0.9, 0.1], random_state=42)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train sample models
    models = {
        'XGBoost': xgb.XGBClassifier(random_state=42, eval_metric='logloss'),
        'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
    }
    
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
    
    return models, X_test_scaled, y_test

def create_confusion_matrices(models, X_test, y_test):
    """Create confusion matrices for all models."""
    print("📊 Creating Confusion Matrices...")
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle('Confusion Matrices - Model Performance Comparison', fontsize=18, fontweight='bold')
    
    for idx, (name, model) in enumerate(models.items()):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        # Create heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Legitimate', 'Fraudulent'],
                    yticklabels=['Legitimate', 'Fraudulent'],
                    ax=axes[idx])
        
        axes[idx].set_title(f'{name}', fontweight='bold')
        axes[idx].set_xlabel('Predicted')
        axes[idx].set_ylabel('Actual')
        
        # Add performance metrics
        tn, fp, fn, tp = cm.ravel()
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        axes[idx].text(0.5, -0.3, f'Accuracy: {accuracy:.3f}\nPrecision: {precision:.3f}\nRecall: {recall:.3f}\nF1: {f1:.3f}', 
                      ha='center', transform=axes[idx].transAxes, 
                      bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('visualizations/confusion_matrices.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Confusion matrices created and saved!")

def create_roc_curves(models, X_test, y_test):
    """Create ROC curves for all models."""
    print("📈 Creating ROC Curves...")
    
    plt.figure(figsize=(12, 8))
    
    for name, model in models.items():
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, linewidth=2, label=f'{name} (AUC = {roc_auc:.3f})')
    
    # Plot diagonal line
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontweight='bold')
    plt.ylabel('True Positive Rate (Sensitivity)', fontweight='bold')
    plt.title('ROC Curves - Model Performance Comparison', fontsize=16, fontweight='bold')
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Add AUC values in text
    plt.text(0.6, 0.2, 'AUC Interpretation:\n• 0.9-1.0: Excellent\n• 0.8-0.9: Good\n• 0.7-0.8: Fair\n• 0.6-0.7: Poor', 
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('visualizations/roc_curves.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ ROC curves created and saved!")

def create_precision_recall_curves(models, X_test, y_test):
    """Create Precision-Recall curves for all models."""
    print("🎯 Creating Precision-Recall Curves...")
    
    plt.figure(figsize=(12, 8))
    
    for name, model in models.items():
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        
        # Calculate F1 score for different thresholds
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        best_f1_idx = np.argmax(f1_scores)
        best_f1 = f1_scores[best_f1_idx]
        
        plt.plot(recall, precision, linewidth=2, 
                label=f'{name} (Best F1 = {best_f1:.3f})')
    
    # Plot baseline
    baseline = len(y_test[y_test == 1]) / len(y_test)
    plt.axhline(y=baseline, color='k', linestyle='--', alpha=0.5, 
                label=f'Baseline (Fraud Rate = {baseline:.3f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall (Sensitivity)', fontweight='bold')
    plt.ylabel('Precision', fontweight='bold')
    plt.title('Precision-Recall Curves - Model Performance Comparison', fontsize=16, fontweight='bold')
    plt.legend(loc="lower left", fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Add interpretation
    plt.text(0.6, 0.2, 'PR Curve Interpretation:\n• Higher curves = Better performance\n• Focus on high recall for fraud detection\n• Balance precision vs recall', 
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('visualizations/precision_recall_curves.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Precision-Recall curves created and saved!")

def create_model_comparison_chart(models, X_test, y_test):
    """Create a comprehensive model comparison chart."""
    print("🏆 Creating Model Comparison Chart...")
    
    # Calculate metrics for all models
    results = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        results[name] = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'F1-Score': f1_score(y_test, y_pred),
            'AUC-ROC': roc_auc_score(y_test, y_pred_proba)
        }
    
    # Create DataFrame
    results_df = pd.DataFrame(results).T
    
    # Create comparison chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    fig.suptitle('Comprehensive Model Performance Comparison', fontsize=18, fontweight='bold')
    
    # Bar chart for key metrics
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']
    x = np.arange(len(metrics))
    width = 0.25
    
    for i, (name, color) in enumerate(zip(results_df.index, ['#2E8B57', '#DC143C', '#4169E1'])):
        values = [results_df.loc[name, metric] for metric in metrics]
        ax1.bar(x + i*width, values, width, label=name, color=color, alpha=0.8)
    
    ax1.set_xlabel('Metrics', fontweight='bold')
    ax1.set_ylabel('Score', fontweight='bold')
    ax1.set_title('Model Performance by Metric', fontweight='bold')
    ax1.set_xticks(x + width)
    ax1.set_xticklabels(metrics)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Heatmap for detailed comparison
    sns.heatmap(results_df, annot=True, fmt='.3f', cmap='RdYlGn', 
                ax=ax2, cbar_kws={'label': 'Score'})
    ax2.set_title('Detailed Performance Matrix', fontweight='bold')
    ax2.set_xlabel('Metrics', fontweight='bold')
    ax2.set_ylabel('Models', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('visualizations/model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Model comparison chart created and saved!")
    return results_df

def create_business_impact_analysis(models, X_test, y_test):
    """Create business impact analysis visualization."""
    print("💼 Creating Business Impact Analysis...")
    
    # Use best model (XGBoost) for business analysis
    best_model = models['XGBoost']
    y_pred = best_model.predict(X_test)
    
    # Calculate business metrics
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    
    # Business impact calculations
    total_transactions = len(y_test)
    fraud_detected = tp
    fraud_missed = fn
    false_alarms = fp
    
    # Estimated financial impact (assuming average transaction amount)
    avg_transaction_amount = 100  # Example value
    fraud_prevented_value = fraud_detected * avg_transaction_amount
    fraud_missed_value = fraud_missed * avg_transaction_amount
    investigation_cost = false_alarms * 50  # Cost per false alarm investigation
    
    net_savings = fraud_prevented_value - investigation_cost
    
    # Create visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Business Impact Analysis - Fraud Detection System', fontsize=18, fontweight='bold')
    
    # 1. Detection Results
    detection_data = [fraud_detected, fraud_missed]
    detection_labels = ['Fraud Detected', 'Fraud Missed']
    colors = ['#2E8B57', '#DC143C']
    
    ax1.pie(detection_data, labels=detection_labels, autopct='%1.1f%%', 
            colors=colors, startangle=90, explode=(0, 0.1))
    ax1.set_title('Fraud Detection Results', fontweight='bold')
    
    # 2. Financial Impact
    financial_data = [fraud_prevented_value, fraud_missed_value, investigation_cost]
    financial_labels = ['Fraud Prevented', 'Fraud Missed', 'Investigation Cost']
    financial_colors = ['#2E8B57', '#DC143C', '#FFD700']
    
    bars = ax2.bar(financial_labels, financial_data, color=financial_colors, alpha=0.8)
    ax2.set_title('Financial Impact ($)', fontweight='bold')
    ax2.set_ylabel('Amount ($)', fontweight='bold')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'${int(height):,}', ha='center', va='bottom', fontweight='bold')
    
    # 3. System Performance Metrics
    performance_metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    performance_values = [
        (tp + tn) / (tp + tn + fp + fn),
        tp / (tp + fp) if (tp + fp) > 0 else 0,
        tp / (tp + fn) if (tp + fn) > 0 else 0,
        2 * (tp / (tp + fp) if (tp + fp) > 0 else 0) * (tp / (tp + fn) if (tp + fn) > 0 else 0) / 
        ((tp / (tp + fp) if (tp + fp) > 0 else 0) + (tp / (tp + fn) if (tp + fn) > 0 else 0)) if 
        ((tp / (tp + fp) if (tp + fp) > 0 else 0) + (tp / (tp + fn) if (tp + fn) > 0 else 0)) > 0 else 0
    ]
    
    bars = ax3.bar(performance_metrics, performance_values, color='#4169E1', alpha=0.8)
    ax3.set_title('System Performance Metrics', fontweight='bold')
    ax3.set_ylabel('Score', fontweight='bold')
    ax3.set_ylim(0, 1)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 4. Summary Statistics
    ax4.axis('off')
    summary_text = f"""
    📊 BUSINESS IMPACT SUMMARY
    
    💰 Financial Impact:
    • Fraud Prevented: ${fraud_prevented_value:,}
    • Fraud Missed: ${fraud_missed_value:,}
    • Investigation Cost: ${investigation_cost:,}
    • Net Savings: ${net_savings:,}
    
    🎯 Detection Performance:
    • Total Transactions: {total_transactions:,}
    • Fraud Detected: {fraud_detected:,} ({tp/(tp+fn)*100:.1f}%)
    • False Alarms: {false_alarms:,}
    
    🏆 System Effectiveness:
    • Accuracy: {(tp+tn)/(tp+tn+fp+fn)*100:.1f}%
    • Precision: {tp/(tp+fp)*100:.1f}%
    • Recall: {tp/(tp+fn)*100:.1f}%
    """
    
    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=12,
              verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('visualizations/business_impact_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Business impact analysis created and saved!")

def create_feature_importance_analysis(models):
    """Create feature importance analysis for tree-based models."""
    print("🔍 Creating Feature Importance Analysis...")
    
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    fig.suptitle('Feature Importance Analysis', fontsize=18, fontweight='bold')
    
    # XGBoost feature importance
    if hasattr(models['XGBoost'], 'feature_importances_'):
        xgb_importance = models['XGBoost'].feature_importances_
        feature_names = [f'Feature_{i}' for i in range(len(xgb_importance))]
        
        # Sort by importance
        sorted_idx = np.argsort(xgb_importance)[::-1]
        top_features = 15  # Show top 15 features
        
        axes[0].barh(range(top_features), xgb_importance[sorted_idx][:top_features])
        axes[0].set_yticks(range(top_features))
        axes[0].set_yticklabels([feature_names[i] for i in sorted_idx][:top_features])
        axes[0].set_xlabel('Importance', fontweight='bold')
        axes[0].set_title('XGBoost Feature Importance (Top 15)', fontweight='bold')
        axes[0].grid(True, alpha=0.3)
    
    # Random Forest feature importance
    if hasattr(models['Random Forest'], 'feature_importances_'):
        rf_importance = models['Random Forest'].feature_importances_
        feature_names = [f'Feature_{i}' for i in range(len(rf_importance))]
        
        # Sort by importance
        sorted_idx = np.argsort(rf_importance)[::-1]
        top_features = 15  # Show top 15 features
        
        axes[1].barh(range(top_features), rf_importance[sorted_idx][:top_features])
        axes[1].set_yticks(range(top_features))
        axes[1].set_yticklabels([feature_names[i] for i in sorted_idx][:top_features])
        axes[1].set_xlabel('Importance', fontweight='bold')
        axes[1].set_title('Random Forest Feature Importance (Top 15)', fontweight='bold')
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('visualizations/feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Feature importance analysis created and saved!")

def main():
    """Main function to create all visualizations."""
    print("🎨 Creating Professional Visualizations for Portfolio...")
    print("=" * 60)
    
    # Setup plotting style
    setup_plotting_style()
    
    # Create visualizations directory
    os.makedirs('visualizations', exist_ok=True)
    
    # Load models and data
    models, X_test, y_test = load_models_and_data()
    
    # Create all visualizations
    create_confusion_matrices(models, X_test, y_test)
    create_roc_curves(models, X_test, y_test)
    create_precision_recall_curves(models, X_test, y_test)
    create_model_comparison_chart(models, X_test, y_test)
    create_business_impact_analysis(models, X_test, y_test)
    create_feature_importance_analysis(models)
    
    print("\n🎉 All Professional Visualizations Created Successfully!")
    print("=" * 60)
    print("📁 Visualizations saved in 'visualizations/' folder:")
    print("   • confusion_matrices.png")
    print("   • roc_curves.png")
    print("   • precision_recall_curves.png")
    print("   • model_comparison.png")
    print("   • business_impact_analysis.png")
    print("   • feature_importance.png")
    print("\n🚀 Your portfolio is now ready with professional charts!")

if __name__ == "__main__":
    main() 