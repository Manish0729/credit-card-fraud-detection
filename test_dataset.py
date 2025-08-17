#!/usr/bin/env python3
"""
Quick script to test dataset loading and display basic information.
Run this after adding your dataset to verify everything works.
"""

import os
import sys

def test_dataset_loading():
    """Test if the dataset can be loaded and display basic info."""
    print("🔍 Testing Dataset Loading...")
    print("=" * 50)
    
    # Check data folder
    data_folder = "data"
    if not os.path.exists(data_folder):
        print(f"❌ {data_folder}/ folder not found!")
        return
    
    # List files in data folder
    data_files = os.listdir(data_folder)
    print(f"📁 Files in {data_folder}/ folder:")
    
    if not data_files:
        print("   ⚠️  No files found - you need to add your dataset!")
        print("\n📥 To add your dataset:")
        print("   1. Download from: https://www.kaggle.com/datasets/kartik2112/fraud-detection")
        print("   2. Place the CSV file in the data/ folder")
        print("   3. Name it: fraud_detection.csv, fraud.csv, or creditcard.csv")
        return
    
    csv_files = [f for f in data_files if f.endswith('.csv')]
    
    if not csv_files:
        print("   ⚠️  No CSV files found!")
        print("   📋 Available files:", data_files)
        return
    
    print(f"   ✅ Found {len(csv_files)} CSV file(s):")
    for csv_file in csv_files:
        file_path = os.path.join(data_folder, csv_file)
        file_size = os.path.getsize(file_path)
        print(f"      • {csv_file} ({file_size:,} bytes)")
    
    # Try to load the first CSV file
    if csv_files:
        test_file = os.path.join(data_folder, csv_files[0])
        print(f"\n🧪 Testing file: {csv_files[0]}")
        
        try:
            # Try to read the first few lines
            with open(test_file, 'r') as f:
                first_line = f.readline().strip()
                second_line = f.readline().strip()
            
            print("✅ File can be read successfully!")
            print(f"📋 First line (headers): {first_line[:100]}...")
            print(f"📋 Second line (data): {second_line[:100]}...")
            
            # Count lines (rough estimate of rows)
            with open(test_file, 'r') as f:
                line_count = sum(1 for _ in f)
            
            print(f"📊 Estimated rows: {line_count:,}")
            
            # Check for target column
            if 'isFraud' in first_line:
                print("🎯 Target column 'isFraud' found!")
            elif 'fraud' in first_line.lower():
                print("🎯 Fraud-related column found!")
            else:
                print("⚠️  No obvious fraud target column found")
                print("   You may need to rename columns or check the dataset structure")
            
        except Exception as e:
            print(f"❌ Error reading file: {e}")
    
    print("\n🎯 Next Steps:")
    if csv_files:
        print("✅ Dataset found! You can now run:")
        print("   python demo.py")
        print("   python src/fraud_detection_main.py")
    else:
        print("📥 Add your dataset first, then run the tests above")

def main():
    """Main function."""
    print("🚀 Credit Card Fraud Detection - Dataset Test")
    print("=" * 60)
    
    test_dataset_loading()
    
    print("\n" + "=" * 60)
    print("🎉 Dataset test completed!")

if __name__ == "__main__":
    main() 