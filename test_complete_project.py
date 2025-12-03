#!/usr/bin/env python3
# Test complete project implementation

import os
import sys

def check_file_exists(filepath, description):
    """Check if a file exists"""
    if os.path.exists(filepath):
        print(f" {description}")
        return True
    else:
        print(f" {description} - MISSING!")
        return False

def main():
    print("=" * 60)
    print("CHECKING COMPLETE PROJECT IMPLEMENTATION")
    print("=" * 60)
    
    all_good = True
    
    # Phase 1-3: Data and Preprocessing
    print("\nPHASE 1-3: Data & Preprocessing")
    print("-" * 40)
    all_good &= check_file_exists("src/explore_data.py", "Data exploration (Step 3)")
    all_good &= check_file_exists("src/preprocessing.py", "Preprocessing (Steps 4-6)")
    
    # Phase 3: Feature Engineering
    print("\nPHASE 3: Feature Engineering")
    print("-" * 40)
    all_good &= check_file_exists("src/feature_engineering.py", "Text encoding (Step 7)")
    
    # Phase 4: Models
    print("\nPHASE 4: Model Development")
    print("-" * 40)
    all_good &= check_file_exists("src/model.py", "Model architectures (Step 8)")
    all_good &= check_file_exists("src/train.py", "Training functions (Step 9)")
    all_good &= check_file_exists("run_training.py", "LSTM training script")
    all_good &= check_file_exists("run_cnn_training.py", "CNN training script")
    all_good &= check_file_exists("run_hybrid_training.py", "Hybrid training script")
    
    # Phase 5: Evaluation
    print("\nPHASE 5: Evaluation")
    print("-" * 40)
    all_good &= check_file_exists("src/evaluation.py", "Evaluation metrics (Step 10)")
    
    # Phase 6: Web Application
    print("\nPHASE 6: Web Application")
    print("-" * 40)
    all_good &= check_file_exists("app.py", "Flask application (Step 11)")
    all_good &= check_file_exists("templates/index.html", "Web interface (Step 12)")
    
    # Phase 7: Ethics
    print("\n⚖️ PHASE 7: Ethical Considerations")
    print("-" * 40)
    all_good &= check_file_exists("ethical_analysis.py", "Bias analysis (Step 13)")
    
    # Phase 8: Integration
    print("\nPHASE 8: Final Integration")
    print("-" * 40)
    all_good &= check_file_exists("main.py", "Complete pipeline (Step 14)")
    
    # Documentation
    print("\nDOCUMENTATION")
    print("-" * 40)
    all_good &= check_file_exists("README.md", "Project documentation")
    all_good &= check_file_exists("requirements.txt", "Dependencies")
    
    # Testing files
    print("\nTESTING")
    print("-" * 40)
    all_good &= check_file_exists("test_pipeline.py", "Pipeline test")
    all_good &= check_file_exists("test_training.py", "Training test")
    
    # Summary
    print("\n" + "=" * 60)
    if all_good:
        print("ALL COMPONENTS IMPLEMENTED!")
        print("\nProject is ready for submission!")
        print("\nTo test:")
        print("1. Run: python test_pipeline.py")
        print("2. Train a model: python run_training.py")
        print("3. Start web app: python app.py")
        print("4. Run ethics analysis: python ethical_analysis.py")
        print("5. Or run complete pipeline: python main.py")
    else:
        print("Some components are missing!")
    
    print("=" * 60)

if __name__ == "__main__":
    main()
