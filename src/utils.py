"""
Utility functions for Credit Card Fraud Detection project.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings('ignore')

def setup_plotting_style():
    """Setup professional plotting style for the project."""
    plt.style.use('default')
    sns.set_palette("husl")
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3

def create_fraud_distribution_plot(df, target_col='isFraud'):
    """Create professional fraud distribution visualization."""
    if target_col not in df.columns:
        print(f"Target column '{target_col}' not found in dataset")
        return
    
    fraud_counts = df[target_col].value_counts()
    fraud_percentage = (fraud_counts / len(df) * 100).round(2)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Pie chart
    colors = ['#2E8B57', '#DC143C']  # SeaGreen, Crimson
    ax1.pie(fraud_counts.values, labels=['Legitimate', 'Fraudulent'], 
            autopct='%1.1f%%', startangle=90, colors=colors, explode=(0, 0.1))
    ax1.set_title('Transaction Distribution', fontsize=16, fontweight='bold')
    
    # Bar chart
    bars = ax2.bar(['Legitimate', 'Fraudulent'], fraud_counts.values, color=colors)
    ax2.set_title('Transaction Counts', fontsize=16, fontweight='bold')
    ax2.set_ylabel('Count')
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{int(height):,}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    # Print statistics
    print(f"📊 Dataset Statistics:")
    print(f"• Total transactions: {len(df):,}")
    print(f"• Legitimate: {fraud_counts[0]:,} ({fraud_percentage[0]}%)")
    print(f"• Fraudulent: {fraud_counts[1]:,} ({fraud_percentage[1]}%)")
    print(f"• Class imbalance ratio: {fraud_counts[0]/fraud_counts[1]:.1f}:1")

def create_correlation_heatmap(df, target_col='isFraud'):
    """Create correlation heatmap for numerical features."""
    numerical_df = df.select_dtypes(include=[np.number])
    
    if target_col not in numerical_df.columns:
        print(f"Target column '{target_col}' not found in numerical columns")
        return
    
    # Calculate correlation matrix
    correlation_matrix = numerical_df.corr()
    
    # Create heatmap
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": .8}, fmt='.2f')
    plt.title('Feature Correlation Matrix', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    # Show top correlations with target
    fraud_correlations = correlation_matrix[target_col].sort_values(ascending=False)
    print(f"\n🎯 Top Features Correlated with {target_col}:")
    print(fraud_correlations.head(10))

def create_model_comparison_plot(results_df):
    """Create comprehensive model comparison visualization."""
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.ravel()
    
    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc', 'training_time']
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC', 'Training Time (s)']
    
    for i, (metric, name) in enumerate(zip(metrics, metric_names)):
        if i < 6:
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

def create_confusion_matrices(trained_models, X_test_scaled, y_test):
    """Create confusion matrices for all models."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    axes = axes.ravel()
    
    for i, (name, model) in enumerate(trained_models.items()):
        if i < 3:
            y_pred = model.predict(X_test_scaled)
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

def create_roc_curves(trained_models, X_test_scaled, y_test):
    """Create ROC curves comparison."""
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

def print_model_performance_summary(results_df):
    """Print comprehensive model performance summary."""
    print("🏆 Model Performance Summary")
    print("=" * 50)
    
    # Find best model for each metric
    for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']:
        best_model = results_df[metric].idxmax()
        best_score = results_df.loc[best_model, metric]
        print(f"• {metric.replace('_', ' ').title()}: {best_model} ({best_score:.4f})")
    
    # Overall best model (F1-score)
    best_overall = results_df['f1_score'].idxmax()
    best_f1 = results_df.loc[best_overall, 'f1_score']
    print(f"\n🥇 Best Overall Model: {best_overall} (F1-Score: {best_f1:.4f})")

def calculate_business_impact(y_test, y_pred, df_original, amount_col='amount'):
    """Calculate business impact of fraud detection."""
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    total_fraud = tp + fn
    fraud_detected = tp
    fraud_missed = fn
    false_alarms = fp
    
    # Calculate financial impact
    if amount_col in df_original.columns:
        avg_fraud_amount = df_original[df_original['isFraud'] == 1][amount_col].mean()
        total_fraud_loss_prevented = fraud_detected * avg_fraud_amount
        total_fraud_loss_missed = fraud_missed * avg_fraud_amount
    else:
        avg_fraud_amount = 1000  # Default estimate
        total_fraud_loss_prevented = fraud_detected * avg_fraud_amount
        total_fraud_loss_missed = fraud_missed * avg_fraud_amount
    
    cost_per_false_alarm = 50  # Estimated cost of investigating false alarm
    total_false_alarm_cost = false_alarms * cost_per_false_alarm
    net_savings = total_fraud_loss_prevented - total_false_alarm_cost
    
    print("💼 Business Impact Analysis")
    print("=" * 40)
    print(f"🔍 Total fraudulent transactions: {total_fraud:,}")
    print(f"✅ Fraudulent transactions detected: {fraud_detected:,} ({fraud_detected/total_fraud*100:.1f}%)")
    print(f"❌ Fraudulent transactions missed: {fraud_missed:,} ({fraud_missed/total_fraud*100:.1f}%)")
    print(f"⚠️  False alarms: {false_alarms:,}")
    print(f"\n💰 Financial Impact (Estimated):")
    print(f"💸 Fraud loss prevented: ${total_fraud_loss_prevented:,.2f}")
    print(f"💸 Fraud loss missed: ${total_fraud_loss_missed:,.2f}")
    print(f"💸 False alarm investigation cost: ${total_false_alarm_cost:,.2f}")
    print(f"💚 Net savings: ${net_savings:,.2f}")
    
    return {
        'fraud_detected': fraud_detected,
        'fraud_missed': fraud_missed,
        'false_alarms': false_alarms,
        'net_savings': net_savings
    } 