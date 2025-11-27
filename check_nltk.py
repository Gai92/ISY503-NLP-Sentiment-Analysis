#!/usr/bin/env python3
# Check what NLTK data we have

import os

project_root = os.path.dirname(os.path.abspath(__file__))
nltk_dir = os.path.join(project_root, "nltk_data")

print("Checking NLTK data...")
print(f"NLTK directory: {nltk_dir}")
print("-" * 40)

# Check if directory exists
if not os.path.exists(nltk_dir):
    print("NLTK directory does not exist!")
    print("Run: python3 fix_nltk.py")
else:
    print("NLTK directory exists")
    
    # Check corpora
    corpora_dir = os.path.join(nltk_dir, "corpora")
    if os.path.exists(corpora_dir):
        print("\nCorpora folder contents:")
        for item in os.listdir(corpora_dir):
            item_path = os.path.join(corpora_dir, item)
            if os.path.isdir(item_path):
                print(f"  [DIR] {item}")
            else:
                print(f"  [FILE] {item}")
    else:
        print("No corpora folder")
    
    # Check stopwords specifically
    stopwords_dir = os.path.join(corpora_dir, "stopwords")
    if os.path.exists(stopwords_dir):
        print("\nStopwords folder contents:")
        files = os.listdir(stopwords_dir)
        print(f"  Found {len(files)} language files")
        if "english" in files:
            print("  [OK] English stopwords found")
        else:
            print("  [MISSING] English stopwords not found")
    else:
        print("\nStopwords folder not found")
    
    # Check tokenizers
    tokenizers_dir = os.path.join(nltk_dir, "tokenizers")
    if os.path.exists(tokenizers_dir):
        print("\nTokenizers folder contents:")
        for item in os.listdir(tokenizers_dir):
            print(f"  {item}")
    else:
        print("\nNo tokenizers folder")

print("\n" + "-" * 40)
print("To fix missing data, run:")
print("  python3 fix_nltk.py")
