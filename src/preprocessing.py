"""
Data preprocessing functions for Credit Card Fraud Detection project.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

def load_and_validate_data(file_paths):
    """
    Load dataset from multiple possible file paths.
    
    Args:
        file_paths (list): List of possible file paths to try
        
    Returns:
        pd.DataFrame: Loaded dataset
    """
    for file_path in file_paths:
        try:
            df = pd.read_csv(file_path)
            print(f"✅ Dataset loaded from {file_path}")
            return df
        except FileNotFoundError:
            continue
    
    # If no file found, create sample data
    print("⚠️  No dataset files found. Creating sample data for demonstration...")
    return create_sample_dataset()

def create_sample_dataset(n_samples=10000):
    """
    Create synthetic dataset similar to credit card fraud data.
    
    Args:
        n_samples (int): Number of samples to create
        
    Returns:
        pd.DataFrame: Synthetic dataset
    """
    np.random.seed(42)
    
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

def clean_data(df):
    """
    Clean the dataset by handling missing values and duplicates.
    
    Args:
        df (pd.DataFrame): Input dataset
        
    Returns:
        pd.DataFrame: Cleaned dataset
    """
    print("🧹 Starting data cleaning...")
    
    # Create a copy for cleaning
    df_clean = df.copy()
    original_shape = df_clean.shape
    
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
        
        # Handle missing values
        for col in missing_df.index:
            if df_clean[col].dtype in ['int64', 'float64']:
                # For numerical columns, fill with median
                median_val = df_clean[col].median()
                df_clean[col].fillna(median_val, inplace=True)
                print(f"✅ Filled missing values in {col} with median: {median_val}")
            else:
                # For categorical columns, fill with mode
                mode_val = df_clean[col].mode()[0]
                df_clean[col].fillna(mode_val, inplace=True)
                print(f"✅ Filled missing values in {col} with mode: {mode_val}")
    else:
        print("✅ No missing values found!")
    
    # Check for duplicates
    duplicates = df_clean.duplicated().sum()
    print(f"\n🔍 Duplicate rows: {duplicates}")
    if duplicates > 0:
        df_clean.drop_duplicates(inplace=True)
        print(f"✅ Removed {duplicates} duplicate rows")
    
    print(f"📊 Dataset shape after cleaning: {original_shape} → {df_clean.shape}")
    return df_clean

def engineer_features(df):
    """
    Create new features to improve model performance.
    
    Args:
        df (pd.DataFrame): Input dataset
        
    Returns:
        pd.DataFrame: Dataset with new features
    """
    print("🔤 Starting feature engineering...")
    
    df_engineered = df.copy()
    
    # Balance change features
    if 'amount' in df_engineered.columns and 'oldbalanceOrg' in df_engineered.columns:
        # Balance change features
        df_engineered['balance_change_orig'] = df_engineered['newbalanceOrig'] - df_engineered['oldbalanceOrg']
        df_engineered['balance_change_dest'] = df_engineered['newbalanceDest'] - df_engineered['oldbalanceDest']
        
        # Amount to balance ratio
        df_engineered['amount_to_balance_ratio'] = df_engineered['amount'] / (df_engineered['oldbalanceOrg'] + 1)
        
        # Balance difference
        df_engineered['balance_difference'] = abs(df_engineered['balance_change_orig']) - abs(df_engineered['balance_change_dest'])
        
        print("✅ Created balance change and ratio features")
    
    # Transaction type features
    if 'type' in df_engineered.columns:
        # Create dummy variables for transaction types
        type_dummies = pd.get_dummies(df_engineered['type'], prefix='type')
        df_engineered = pd.concat([df_engineered, type_dummies], axis=1)
        print("✅ Created transaction type dummy variables")
    
    # Time-based features
    if 'step' in df_engineered.columns:
        # Create time bins
        df_engineered['time_bin'] = pd.cut(df_engineered['step'], bins=10, labels=False)
        print("✅ Created time bin features")
    
    # Statistical features
    if 'amount' in df_engineered.columns:
        # Amount statistics
        df_engineered['amount_log'] = np.log1p(df_engineered['amount'])
        df_engineered['amount_squared'] = df_engineered['amount'] ** 2
        print("✅ Created amount transformation features")
    
    print(f"📊 Dataset shape after feature engineering: {df_engineered.shape}")
    return df_engineered

def encode_categorical_variables(df, exclude_cols=None):
    """
    Encode categorical variables using Label Encoding.
    
    Args:
        df (pd.DataFrame): Input dataset
        exclude_cols (list): Columns to exclude from encoding
        
    Returns:
        tuple: (encoded_dataset, label_encoders_dict)
    """
    print("🔤 Starting categorical variable encoding...")
    
    if exclude_cols is None:
        exclude_cols = []
    
    df_encoded = df.copy()
    label_encoders = {}
    
    # Find categorical columns
    categorical_cols = df_encoded.select_dtypes(include=['object']).columns
    categorical_cols = [col for col in categorical_cols if col not in exclude_cols]
    
    for col in categorical_cols:
        if df_encoded[col].nunique() > 1:  # Only encode if more than 1 unique value
            le = LabelEncoder()
            df_encoded[f'{col}_encoded'] = le.fit_transform(df_encoded[col])
            label_encoders[col] = le
            print(f"✅ Encoded {col} with {df_encoded[col].nunique()} unique values")
    
    # Remove original categorical columns (keep encoded versions)
    df_encoded.drop(columns=categorical_cols, inplace=True)
    
    print(f"📊 Dataset shape after encoding: {df_encoded.shape}")
    return df_encoded, label_encoders

def prepare_features_and_target(df, target_col='isFraud', exclude_cols=None):
    """
    Prepare features and target variables for machine learning.
    
    Args:
        df (pd.DataFrame): Input dataset
        target_col (str): Name of target column
        exclude_cols (list): Columns to exclude from features
        
    Returns:
        tuple: (features, target)
    """
    print("🎯 Preparing features and target...")
    
    if exclude_cols is None:
        exclude_cols = []
    
    # Ensure target column exists
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset")
    
    # Prepare features
    feature_cols = [col for col in df.columns if col not in [target_col] + exclude_cols]
    X = df[feature_cols]
    y = df[target_col]
    
    print(f"🔤 Features shape: {X.shape}")
    print(f"🎯 Target shape: {y.shape}")
    print(f"📋 Feature columns: {list(X.columns)}")
    
    return X, y

def handle_class_imbalance(X, y, method='smote', random_state=42):
    """
    Handle class imbalance using various techniques.
    
    Args:
        X (pd.DataFrame): Feature matrix
        y (pd.Series): Target variable
        method (str): Method to use ('smote', 'undersample', 'combine')
        random_state (int): Random seed
        
    Returns:
        tuple: (balanced_features, balanced_target)
    """
    print(f"⚖️ Handling class imbalance using {method.upper()}...")
    
    # Check current class distribution
    class_counts = y.value_counts()
    print(f"📊 Current class distribution:")
    for class_val, count in class_counts.items():
        percentage = (count / len(y) * 100)
        print(f"   Class {class_val}: {count:,} ({percentage:.2f}%)")
    
    if method.lower() == 'smote':
        smote = SMOTE(random_state=random_state, k_neighbors=5)
        X_balanced, y_balanced = smote.fit_resample(X, y)
    elif method.lower() == 'undersample':
        from imblearn.under_sampling import RandomUnderSampler
        undersampler = RandomUnderSampler(random_state=random_state)
        X_balanced, y_balanced = undersampler.fit_resample(X, y)
    elif method.lower() == 'combine':
        from imblearn.combine import SMOTEENN
        smoteenn = SMOTEENN(random_state=random_state)
        X_balanced, y_balanced = smoteenn.fit_resample(X, y)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Check balanced class distribution
    balanced_class_counts = y_balanced.value_counts()
    print(f"📊 Balanced class distribution:")
    for class_val, count in balanced_class_counts.items():
        percentage = (count / len(y_balanced) * 100)
        print(f"   Class {class_val}: {count:,} ({percentage:.2f}%)")
    
    return X_balanced, y_balanced

def scale_features(X_train, X_test, method='standard'):
    """
    Scale features using specified method.
    
    Args:
        X_train (pd.DataFrame): Training features
        X_test (pd.DataFrame): Test features
        method (str): Scaling method ('standard', 'minmax', 'robust')
        
    Returns:
        tuple: (scaled_training_features, scaled_test_features, scaler)
    """
    print(f"📏 Scaling features using {method} scaling...")
    
    if method.lower() == 'standard':
        scaler = StandardScaler()
    elif method.lower() == 'minmax':
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
    elif method.lower() == 'robust':
        from sklearn.preprocessing import RobustScaler
        scaler = RobustScaler()
    else:
        raise ValueError(f"Unknown scaling method: {method}")
    
    # Fit scaler on training data and transform both
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("✅ Feature scaling completed!")
    return X_train_scaled, X_test_scaled, scaler

def split_and_prepare_data(df, target_col='isFraud', test_size=0.2, random_state=42):
    """
    Complete data preparation pipeline.
    
    Args:
        df (pd.DataFrame): Input dataset
        target_col (str): Target column name
        test_size (float): Proportion of test set
        random_state (int): Random seed
        
    Returns:
        tuple: (X_train_scaled, X_test_scaled, y_train, y_test, scaler, label_encoders)
    """
    print("🚀 Starting complete data preparation pipeline...")
    print("=" * 60)
    
    # Clean data
    df_clean = clean_data(df)
    
    # Engineer features
    df_engineered = engineer_features(df_clean)
    
    # Encode categorical variables
    df_encoded, label_encoders = encode_categorical_variables(df_engineered)
    
    # Prepare features and target
    X, y = prepare_features_and_target(df_encoded, target_col)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"\n📊 Data split:")
    print(f"   Training set: {X_train.shape[0]:,} samples")
    print(f"   Test set: {X_test.shape[0]:,} samples")
    
    # Handle class imbalance in training data only
    X_train_balanced, y_train_balanced = handle_class_imbalance(X_train, y_train)
    
    # Scale features
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train_balanced, X_test)
    
    print("\n✅ Data preparation completed successfully!")
    return X_train_scaled, X_test_scaled, y_train_balanced, y_test, scaler, label_encoders 