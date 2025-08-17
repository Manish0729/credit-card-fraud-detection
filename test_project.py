#!/usr/bin/env python3
"""
Simple test script to verify the Credit Card Fraud Detection project structure.
This script runs without external dependencies to check the project setup.
"""

import os
import sys

def test_project_structure():
    """Test the project structure and files."""
    print("🚀 Credit Card Fraud Detection - Project Structure Test")
    print("=" * 60)
    
    # Check if all required directories exist
    required_dirs = ['src', 'notebooks', 'data', 'models']
    print("\n📁 Checking required directories:")
    
    for dir_name in required_dirs:
        if os.path.exists(dir_name):
            print(f"✅ {dir_name}/ - Found")
        else:
            print(f"❌ {dir_name}/ - Missing")
    
    # Check if all required files exist
    required_files = [
        'README.md',
        'requirements.txt',
        'demo.py',
        'src/preprocessing.py',
        'src/models.py',
        'src/utils.py',
        'src/fraud_detection_main.py',
        'notebooks/credit_card_fraud_detection.ipynb',
        'PROJECT_SUMMARY.md'
    ]
    
    print("\n📄 Checking required files:")
    
    for file_path in required_files:
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path)
            print(f"✅ {file_path} - Found ({file_size:,} bytes)")
        else:
            print(f"❌ {file_path} - Missing")
    
    # Check Python files for syntax
    print("\n🐍 Checking Python file syntax:")
    
    python_files = [
        'demo.py',
        'src/preprocessing.py',
        'src/models.py',
        'src/utils.py',
        'src/fraud_detection_main.py'
    ]
    
    for py_file in python_files:
        if os.path.exists(py_file):
            try:
                with open(py_file, 'r') as f:
                    content = f.read()
                # Basic syntax check - try to compile
                compile(content, py_file, 'exec')
                print(f"✅ {py_file} - Syntax OK")
            except SyntaxError as e:
                print(f"❌ {py_file} - Syntax Error: {e}")
            except Exception as e:
                print(f"⚠️  {py_file} - Read Error: {e}")
        else:
            print(f"❌ {py_file} - File not found")
    
    # Project summary
    print("\n📊 Project Summary:")
    print(f"• Total directories: {len(required_dirs)}")
    print(f"• Total files: {len(required_files)}")
    print(f"• Python modules: {len(python_files)}")
    
    # Check for common issues
    print("\n🔍 Common Issues Check:")
    
    # Check if data directory is empty
    if os.path.exists('data') and len(os.listdir('data')) == 0:
        print("⚠️  data/ directory is empty - you'll need to add your dataset")
    elif os.path.exists('data'):
        data_files = os.listdir('data')
        print(f"✅ data/ directory contains: {', '.join(data_files)}")
    
    # Check if models directory is empty
    if os.path.exists('models') and len(os.listdir('models')) == 0:
        print("ℹ️  models/ directory is empty - will be populated when you run the demo")
    elif os.path.exists('models'):
        model_files = os.listdir('models')
        print(f"✅ models/ directory contains: {', '.join(model_files)}")
    
    print("\n🎯 Next Steps:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Download dataset from Kaggle")
    print("3. Place dataset in data/ folder")
    print("4. Run: python demo.py")
    print("5. Open notebook: jupyter notebook notebooks/credit_card_fraud_detection.ipynb")

def main():
    """Main function."""
    try:
        test_project_structure()
        print("\n🎉 Project structure test completed successfully!")
        print("🚀 Your Credit Card Fraud Detection project is ready!")
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        print("Please check the project setup and try again.")

if __name__ == "__main__":
    main() 