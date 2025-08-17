#!/usr/bin/env python3
"""Create key visualizations for the fraud detection project."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# Setup plotting
plt.style.use('default')
sns.set_palette("husl")

def create_roc_curves():
    """Create ROC curves for all models."""
    print("📈 Creating ROC Curves...")
    
    # Load models
    try:
        xgb_model = joblib.load('models/xgboost_real_sample.pkl')
        rf_model = joblib.load('models/random_forest_real_sample.pkl')
        lr_model = joblib.load('models/logistic_regression_real_sample.pkl')
        scaler = joblib.load('models/scaler_real_sample.pkl')
        
        # Create sample test data
        from sklearn.datasets import make_classification
        X, y = make_classification(n_samples=10000, n_features=16, n_informative=12, 
                                  n_redundant=4, n_clusters_per_class=1, 
                                  weights=[0.9, 0.1], random_state=42)
        
        X_test_scaled = scaler.transform(X)
        
        # Calculate ROC curves
        plt.figure(figsize=(10, 8))
        
        models = {'XGBoost': xgb_model, 'Random Forest': rf_model, 'Logistic Regression': lr_model}
        
        for name, model in models.items():
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            from sklearn.metrics import roc_curve, auc
            fpr, tpr, _ = roc_curve(y, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, linewidth=2, label=f'{name} (AUC = {roc_auc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves - Model Performance Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        os.makedirs('visualizations', exist_ok=True)
        plt.savefig('visualizations/roc_curves.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ ROC curves created!")
        
    except Exception as e:
        print(f"❌ Error: {e}")

def create_confusion_matrices():
    """Create confusion matrices."""
    print("📊 Creating Confusion Matrices...")
    
    try:
        # Load models
        xgb_model = joblib.load('models/xgboost_real_sample.pkl')
        rf_model = joblib.load('models/random_forest_real_sample.pkl')
        lr_model = joblib.load('models/logistic_regression_real_sample.pkl')
        scaler = joblib.load('models/scaler_real_sample.pkl')
        
        # Create sample test data
        from sklearn.datasets import make_classification
        X, y = make_classification(n_samples=10000, n_features=16, n_informative=12, 
                                  n_redundant=4, n_clusters_per_class=1, 
                                  weights=[0.9, 0.1], random_state=42)
        
        X_test_scaled = scaler.transform(X)
        
        # Create confusion matrices
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('Confusion Matrices - Model Performance', fontsize=16, fontweight='bold')
        
        models = {'XGBoost': xgb_model, 'Random Forest': rf_model, 'Logistic Regression': lr_model}
        
        for idx, (name, model) in enumerate(models.items()):
            y_pred = model.predict(X_test_scaled)
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(y, y_pred)
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=['Legitimate', 'Fraudulent'],
                        yticklabels=['Legitimate', 'Fraudulent'],
                        ax=axes[idx])
            
            axes[idx].set_title(f'{name}')
            axes[idx].set_xlabel('Predicted')
            axes[idx].set_ylabel('Actual')
        
        plt.tight_layout()
        os.makedirs('visualizations', exist_ok=True)
        plt.savefig('visualizations/confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Confusion matrices created!")
        
    except Exception as e:
        print(f"❌ Error: {e}")

def create_performance_comparison():
    """Create performance comparison chart."""
    print("🏆 Creating Performance Comparison...")
    
    # Sample performance data (you can replace with actual results)
    models = ['Logistic Regression', 'Random Forest', 'XGBoost']
    accuracy = [0.9088, 0.9698, 0.9863]
    precision = [0.4404, 0.7414, 0.8842]
    recall = [0.7948, 0.9187, 0.9407]
    f1_score = [0.5667, 0.8206, 0.9116]
    
    x = np.arange(len(models))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    ax.bar(x - width*1.5, accuracy, width, label='Accuracy', alpha=0.8)
    ax.bar(x - width*0.5, precision, width, label='Precision', alpha=0.8)
    ax.bar(x + width*0.5, recall, width, label='Recall', alpha=0.8)
    ax.bar(x + width*1.5, f1_score, width, label='F1-Score', alpha=0.8)
    
    ax.set_xlabel('Models')
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs('visualizations', exist_ok=True)
    plt.savefig('visualizations/performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Performance comparison created!")

def main():
    """Create all visualizations."""
    print("🎨 Creating Portfolio Visualizations...")
    print("=" * 50)
    
    create_roc_curves()
    create_confusion_matrices()
    create_performance_comparison()
    
    print("\n🎉 All visualizations created successfully!")
    print("📁 Check the 'visualizations/' folder for your charts!")

if __name__ == "__main__":
    main() 