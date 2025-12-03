#!/usr/bin/env python3
# run cnn training

import sys
import os

# add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from train import train_cnn

if __name__ == "__main__":
    print("=" * 50)
    print("CNN Training Script")
    print("=" * 50)
    
    try:
        model, history = train_cnn()
        print("\nCNN training finished successfully!")
    except Exception as e:
        print(f"\nError during training: {e}")
        print("\nMake sure you have:")
        print("- Data files in data/ folder")
        print("- All dependencies installed")
        print("- Run: pip install -r requirements.txt")
