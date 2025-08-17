#!/usr/bin/env python3
"""
Demo script for Credit Card Fraud Detection project.
This script demonstrates the complete pipeline from data loading to model evaluation.
"""

import sys
import os

# Add src directory to path
sys.path.append('src')

from preprocessing import load_and_validate_data, split_and_prepare_data
from models import train_all_models, save_models, create_results_dataframe, print_model_summary
from models import create_comprehensive_evaluation_plots, get_best_model
from utils import create_fraud_distribution_plot, create_correlation_heatmap, calculate_business_impact
import joblib

def main():
    """Run the complete fraud detection pipeline."""
    print("🚀 Credit Card Fraud Detection - Demo Pipeline")
    print("=" * 70)
    
    # Step 1: Load data
    print("\n📥 Step 1: Loading Data")
    print("-" * 40)
    
    file_paths = [
        'data/fraud_detection.csv',
        'data/fraud.csv',
        'data/creditcard.csv'
    ]
    
    df = load_and_validate_data(file_paths)
    print(f"✅ Dataset loaded with shape: {df.shape}")
    
    # Step 2: Exploratory Data Analysis
    print("\n🔍 Step 2: Exploratory Data Analysis")
    print("-" * 40)
    
    # Create fraud distribution plot
    if 'isFraud' in df.columns:
        create_fraud_distribution_plot(df)
        
        # Create correlation heatmap
        create_correlation_heatmap(df)
    
    # Step 3: Data Preprocessing
    print("\n🔧 Step 3: Data Preprocessing")
    print("-" * 40)
    
    try:
        X_train_scaled, X_test_scaled, y_train_balanced, y_test, scaler, label_encoders = \
            split_and_prepare_data(df, target_col='isFraud')
        
        print("✅ Data preprocessing completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during data preprocessing: {e}")
        return
    
    # Step 4: Model Training
    print("\n🤖 Step 4: Model Training")
    print("-" * 40)
    
    try:
        results, trained_models = train_all_models(
            X_train_scaled, y_train_balanced, X_test_scaled, y_test
        )
        
        print("✅ Model training completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during model training: {e}")
        return
    
    # Step 5: Model Evaluation
    print("\n📊 Step 5: Model Evaluation")
    print("-" * 40)
    
    # Create results DataFrame
    results_df = create_results_dataframe(results)
    
    # Print model summary
    print_model_summary(results_df)
    
    # Create evaluation plots
    try:
        create_comprehensive_evaluation_plots(results_df, trained_models, X_test_scaled, y_test)
        print("✅ Evaluation visualizations created successfully!")
    except Exception as e:
        print(f"⚠️  Warning: Some visualizations failed: {e}")
    
    # Step 6: Save Models
    print("\n💾 Step 6: Saving Models")
    print("-" * 40)
    
    try:
        save_models(trained_models, scaler, label_encoders)
        print("✅ Models saved successfully!")
    except Exception as e:
        print(f"❌ Error saving models: {e}")
    
    # Step 7: Business Impact Analysis
    print("\n💼 Step 7: Business Impact Analysis")
    print("-" * 40)
    
    try:
        # Get best model
        best_model_name, best_model, best_score = get_best_model(results_df, trained_models)
        
        # Make predictions with best model
        y_pred_best = best_model.predict(X_test_scaled)
        
        # Calculate business impact
        business_impact = calculate_business_impact(y_test, y_pred_best, df)
        
        print(f"\n🏆 Best Model: {best_model_name}")
        print(f"📊 F1-Score: {best_score:.4f}")
        
    except Exception as e:
        print(f"⚠️  Warning: Business impact analysis failed: {e}")
    
    # Step 8: Final Summary
    print("\n🎉 Step 8: Final Summary")
    print("-" * 40)
    
    print("✅ Credit Card Fraud Detection pipeline completed successfully!")
    print(f"📊 Total models trained: {len(trained_models)}")
    print(f"🔤 Features used: {X_train_scaled.shape[1]}")
    print(f"📈 Training samples: {X_train_scaled.shape[0]:,}")
    print(f"🧪 Test samples: {X_test_scaled.shape[0]:,}")
    
    if 'best_model_name' in locals():
        print(f"🥇 Best performing model: {best_model_name}")
        print(f"📊 Best F1-Score: {best_score:.4f}")
    
    print("\n📁 Project files created:")
    print("   • models/ - Saved ML models and preprocessing objects")
    print("   • src/ - Source code modules")
    print("   • README.md - Project documentation")
    print("   • requirements.txt - Python dependencies")
    
    print("\n🚀 Next steps:")
    print("   1. Download the real dataset from Kaggle")
    print("   2. Place it in the data/ folder")
    print("   3. Run this script again for real data analysis")
    print("   4. Customize models and parameters as needed")

if __name__ == "__main__":
    main() 